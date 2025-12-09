# **CMSC320 Checkpoint 2: Leah Brennan, Priyanshi Patel, Reuben Puthumana, Maya Cleveland**

# Data Preprocessing

### (a) Import


```python
import pandas as pd

from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
from scipy.stats import ttest_ind

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv('student_habits_performance.csv')
display(df)
```



  <div id="df-042daaf3-90d8-4cad-a137-dc1d715ad9ce" class="colab-df-container">
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
      <th>student_id</th>
      <th>age</th>
      <th>gender</th>
      <th>study_hours_per_day</th>
      <th>social_media_hours</th>
      <th>netflix_hours</th>
      <th>part_time_job</th>
      <th>attendance_percentage</th>
      <th>sleep_hours</th>
      <th>diet_quality</th>
      <th>exercise_frequency</th>
      <th>parental_education_level</th>
      <th>internet_quality</th>
      <th>mental_health_rating</th>
      <th>extracurricular_participation</th>
      <th>exam_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S1000</td>
      <td>23</td>
      <td>Female</td>
      <td>0.0</td>
      <td>1.2</td>
      <td>1.1</td>
      <td>No</td>
      <td>85.0</td>
      <td>8.0</td>
      <td>Fair</td>
      <td>6</td>
      <td>Master</td>
      <td>Average</td>
      <td>8</td>
      <td>Yes</td>
      <td>56.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S1001</td>
      <td>20</td>
      <td>Female</td>
      <td>6.9</td>
      <td>2.8</td>
      <td>2.3</td>
      <td>No</td>
      <td>97.3</td>
      <td>4.6</td>
      <td>Good</td>
      <td>6</td>
      <td>High School</td>
      <td>Average</td>
      <td>8</td>
      <td>No</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S1002</td>
      <td>21</td>
      <td>Male</td>
      <td>1.4</td>
      <td>3.1</td>
      <td>1.3</td>
      <td>No</td>
      <td>94.8</td>
      <td>8.0</td>
      <td>Poor</td>
      <td>1</td>
      <td>High School</td>
      <td>Poor</td>
      <td>1</td>
      <td>No</td>
      <td>34.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S1003</td>
      <td>23</td>
      <td>Female</td>
      <td>1.0</td>
      <td>3.9</td>
      <td>1.0</td>
      <td>No</td>
      <td>71.0</td>
      <td>9.2</td>
      <td>Poor</td>
      <td>4</td>
      <td>Master</td>
      <td>Good</td>
      <td>1</td>
      <td>Yes</td>
      <td>26.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S1004</td>
      <td>19</td>
      <td>Female</td>
      <td>5.0</td>
      <td>4.4</td>
      <td>0.5</td>
      <td>No</td>
      <td>90.9</td>
      <td>4.9</td>
      <td>Fair</td>
      <td>3</td>
      <td>Master</td>
      <td>Good</td>
      <td>1</td>
      <td>No</td>
      <td>66.4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>S1995</td>
      <td>21</td>
      <td>Female</td>
      <td>2.6</td>
      <td>0.5</td>
      <td>1.6</td>
      <td>No</td>
      <td>77.0</td>
      <td>7.5</td>
      <td>Fair</td>
      <td>2</td>
      <td>High School</td>
      <td>Good</td>
      <td>6</td>
      <td>Yes</td>
      <td>76.1</td>
    </tr>
    <tr>
      <th>996</th>
      <td>S1996</td>
      <td>17</td>
      <td>Female</td>
      <td>2.9</td>
      <td>1.0</td>
      <td>2.4</td>
      <td>Yes</td>
      <td>86.0</td>
      <td>6.8</td>
      <td>Poor</td>
      <td>1</td>
      <td>High School</td>
      <td>Average</td>
      <td>6</td>
      <td>Yes</td>
      <td>65.9</td>
    </tr>
    <tr>
      <th>997</th>
      <td>S1997</td>
      <td>20</td>
      <td>Male</td>
      <td>3.0</td>
      <td>2.6</td>
      <td>1.3</td>
      <td>No</td>
      <td>61.9</td>
      <td>6.5</td>
      <td>Good</td>
      <td>5</td>
      <td>Bachelor</td>
      <td>Good</td>
      <td>9</td>
      <td>Yes</td>
      <td>64.4</td>
    </tr>
    <tr>
      <th>998</th>
      <td>S1998</td>
      <td>24</td>
      <td>Male</td>
      <td>5.4</td>
      <td>4.1</td>
      <td>1.1</td>
      <td>Yes</td>
      <td>100.0</td>
      <td>7.6</td>
      <td>Fair</td>
      <td>0</td>
      <td>Bachelor</td>
      <td>Average</td>
      <td>1</td>
      <td>No</td>
      <td>69.7</td>
    </tr>
    <tr>
      <th>999</th>
      <td>S1999</td>
      <td>19</td>
      <td>Female</td>
      <td>4.3</td>
      <td>2.9</td>
      <td>1.9</td>
      <td>No</td>
      <td>89.4</td>
      <td>7.1</td>
      <td>Good</td>
      <td>2</td>
      <td>Bachelor</td>
      <td>Average</td>
      <td>8</td>
      <td>No</td>
      <td>74.9</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 16 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-042daaf3-90d8-4cad-a137-dc1d715ad9ce')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-042daaf3-90d8-4cad-a137-dc1d715ad9ce button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-042daaf3-90d8-4cad-a137-dc1d715ad9ce');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-e9dcb408-9946-453b-93a6-108c41ac1bd7">
      <button class="colab-df-quickchart" onclick="quickchart('df-e9dcb408-9946-453b-93a6-108c41ac1bd7')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-e9dcb408-9946-453b-93a6-108c41ac1bd7 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_81f2184b-35d3-49bf-80e9-c5fce018c465">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_81f2184b-35d3-49bf-80e9-c5fce018c465 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



### (b) Parse


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 16 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   student_id                     1000 non-null   object 
     1   age                            1000 non-null   int64  
     2   gender                         1000 non-null   object 
     3   study_hours_per_day            1000 non-null   float64
     4   social_media_hours             1000 non-null   float64
     5   netflix_hours                  1000 non-null   float64
     6   part_time_job                  1000 non-null   object 
     7   attendance_percentage          1000 non-null   float64
     8   sleep_hours                    1000 non-null   float64
     9   diet_quality                   1000 non-null   object 
     10  exercise_frequency             1000 non-null   int64  
     11  parental_education_level       909 non-null    object 
     12  internet_quality               1000 non-null   object 
     13  mental_health_rating           1000 non-null   int64  
     14  extracurricular_participation  1000 non-null   object 
     15  exam_score                     1000 non-null   float64
    dtypes: float64(6), int64(3), object(7)
    memory usage: 125.1+ KB



```python
df['part_time_job'] = df['part_time_job'].map({'Yes': True, 'No': False})
df['extracurricular_participation'] = df['extracurricular_participation'].map({'Yes': True, 'No': False})
df['gender'] = df['gender'].astype('category')
df['diet_quality'] = df['diet_quality'].astype('category')
df['parental_education_level'] = df['parental_education_level'].astype('category')
df['internet_quality'] = df['internet_quality'].astype('category')
df.info()
display(df)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 16 columns):
     #   Column                         Non-Null Count  Dtype   
    ---  ------                         --------------  -----   
     0   student_id                     1000 non-null   object  
     1   age                            1000 non-null   int64   
     2   gender                         1000 non-null   category
     3   study_hours_per_day            1000 non-null   float64 
     4   social_media_hours             1000 non-null   float64 
     5   netflix_hours                  1000 non-null   float64 
     6   part_time_job                  1000 non-null   bool    
     7   attendance_percentage          1000 non-null   float64 
     8   sleep_hours                    1000 non-null   float64 
     9   diet_quality                   1000 non-null   category
     10  exercise_frequency             1000 non-null   int64   
     11  parental_education_level       909 non-null    category
     12  internet_quality               1000 non-null   category
     13  mental_health_rating           1000 non-null   int64   
     14  extracurricular_participation  1000 non-null   bool    
     15  exam_score                     1000 non-null   float64 
    dtypes: bool(2), category(4), float64(6), int64(3), object(1)
    memory usage: 84.6+ KB




  <div id="df-86a0a141-2197-49e3-b317-7bd85241dc57" class="colab-df-container">
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
      <th>student_id</th>
      <th>age</th>
      <th>gender</th>
      <th>study_hours_per_day</th>
      <th>social_media_hours</th>
      <th>netflix_hours</th>
      <th>part_time_job</th>
      <th>attendance_percentage</th>
      <th>sleep_hours</th>
      <th>diet_quality</th>
      <th>exercise_frequency</th>
      <th>parental_education_level</th>
      <th>internet_quality</th>
      <th>mental_health_rating</th>
      <th>extracurricular_participation</th>
      <th>exam_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S1000</td>
      <td>23</td>
      <td>Female</td>
      <td>0.0</td>
      <td>1.2</td>
      <td>1.1</td>
      <td>False</td>
      <td>85.0</td>
      <td>8.0</td>
      <td>Fair</td>
      <td>6</td>
      <td>Master</td>
      <td>Average</td>
      <td>8</td>
      <td>True</td>
      <td>56.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>S1001</td>
      <td>20</td>
      <td>Female</td>
      <td>6.9</td>
      <td>2.8</td>
      <td>2.3</td>
      <td>False</td>
      <td>97.3</td>
      <td>4.6</td>
      <td>Good</td>
      <td>6</td>
      <td>High School</td>
      <td>Average</td>
      <td>8</td>
      <td>False</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S1002</td>
      <td>21</td>
      <td>Male</td>
      <td>1.4</td>
      <td>3.1</td>
      <td>1.3</td>
      <td>False</td>
      <td>94.8</td>
      <td>8.0</td>
      <td>Poor</td>
      <td>1</td>
      <td>High School</td>
      <td>Poor</td>
      <td>1</td>
      <td>False</td>
      <td>34.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S1003</td>
      <td>23</td>
      <td>Female</td>
      <td>1.0</td>
      <td>3.9</td>
      <td>1.0</td>
      <td>False</td>
      <td>71.0</td>
      <td>9.2</td>
      <td>Poor</td>
      <td>4</td>
      <td>Master</td>
      <td>Good</td>
      <td>1</td>
      <td>True</td>
      <td>26.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S1004</td>
      <td>19</td>
      <td>Female</td>
      <td>5.0</td>
      <td>4.4</td>
      <td>0.5</td>
      <td>False</td>
      <td>90.9</td>
      <td>4.9</td>
      <td>Fair</td>
      <td>3</td>
      <td>Master</td>
      <td>Good</td>
      <td>1</td>
      <td>False</td>
      <td>66.4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>S1995</td>
      <td>21</td>
      <td>Female</td>
      <td>2.6</td>
      <td>0.5</td>
      <td>1.6</td>
      <td>False</td>
      <td>77.0</td>
      <td>7.5</td>
      <td>Fair</td>
      <td>2</td>
      <td>High School</td>
      <td>Good</td>
      <td>6</td>
      <td>True</td>
      <td>76.1</td>
    </tr>
    <tr>
      <th>996</th>
      <td>S1996</td>
      <td>17</td>
      <td>Female</td>
      <td>2.9</td>
      <td>1.0</td>
      <td>2.4</td>
      <td>True</td>
      <td>86.0</td>
      <td>6.8</td>
      <td>Poor</td>
      <td>1</td>
      <td>High School</td>
      <td>Average</td>
      <td>6</td>
      <td>True</td>
      <td>65.9</td>
    </tr>
    <tr>
      <th>997</th>
      <td>S1997</td>
      <td>20</td>
      <td>Male</td>
      <td>3.0</td>
      <td>2.6</td>
      <td>1.3</td>
      <td>False</td>
      <td>61.9</td>
      <td>6.5</td>
      <td>Good</td>
      <td>5</td>
      <td>Bachelor</td>
      <td>Good</td>
      <td>9</td>
      <td>True</td>
      <td>64.4</td>
    </tr>
    <tr>
      <th>998</th>
      <td>S1998</td>
      <td>24</td>
      <td>Male</td>
      <td>5.4</td>
      <td>4.1</td>
      <td>1.1</td>
      <td>True</td>
      <td>100.0</td>
      <td>7.6</td>
      <td>Fair</td>
      <td>0</td>
      <td>Bachelor</td>
      <td>Average</td>
      <td>1</td>
      <td>False</td>
      <td>69.7</td>
    </tr>
    <tr>
      <th>999</th>
      <td>S1999</td>
      <td>19</td>
      <td>Female</td>
      <td>4.3</td>
      <td>2.9</td>
      <td>1.9</td>
      <td>False</td>
      <td>89.4</td>
      <td>7.1</td>
      <td>Good</td>
      <td>2</td>
      <td>Bachelor</td>
      <td>Average</td>
      <td>8</td>
      <td>False</td>
      <td>74.9</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 16 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-86a0a141-2197-49e3-b317-7bd85241dc57')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-86a0a141-2197-49e3-b317-7bd85241dc57 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-86a0a141-2197-49e3-b317-7bd85241dc57');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-49b4416e-6f59-46b7-84b1-4bec609e2435">
      <button class="colab-df-quickchart" onclick="quickchart('df-49b4416e-6f59-46b7-84b1-4bec609e2435')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-49b4416e-6f59-46b7-84b1-4bec609e2435 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_174233a4-264f-449e-8409-4b32b61a0c7b">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_174233a4-264f-449e-8409-4b32b61a0c7b button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



### (c) Organize


```python
df = df.set_index('student_id')
display(df)
```



  <div id="df-aafa1f92-e582-49f0-89e3-17c4ae38dd5c" class="colab-df-container">
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
      <th>gender</th>
      <th>study_hours_per_day</th>
      <th>social_media_hours</th>
      <th>netflix_hours</th>
      <th>part_time_job</th>
      <th>attendance_percentage</th>
      <th>sleep_hours</th>
      <th>diet_quality</th>
      <th>exercise_frequency</th>
      <th>parental_education_level</th>
      <th>internet_quality</th>
      <th>mental_health_rating</th>
      <th>extracurricular_participation</th>
      <th>exam_score</th>
    </tr>
    <tr>
      <th>student_id</th>
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
      <th>S1000</th>
      <td>23</td>
      <td>Female</td>
      <td>0.0</td>
      <td>1.2</td>
      <td>1.1</td>
      <td>False</td>
      <td>85.0</td>
      <td>8.0</td>
      <td>Fair</td>
      <td>6</td>
      <td>Master</td>
      <td>Average</td>
      <td>8</td>
      <td>True</td>
      <td>56.2</td>
    </tr>
    <tr>
      <th>S1001</th>
      <td>20</td>
      <td>Female</td>
      <td>6.9</td>
      <td>2.8</td>
      <td>2.3</td>
      <td>False</td>
      <td>97.3</td>
      <td>4.6</td>
      <td>Good</td>
      <td>6</td>
      <td>High School</td>
      <td>Average</td>
      <td>8</td>
      <td>False</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>S1002</th>
      <td>21</td>
      <td>Male</td>
      <td>1.4</td>
      <td>3.1</td>
      <td>1.3</td>
      <td>False</td>
      <td>94.8</td>
      <td>8.0</td>
      <td>Poor</td>
      <td>1</td>
      <td>High School</td>
      <td>Poor</td>
      <td>1</td>
      <td>False</td>
      <td>34.3</td>
    </tr>
    <tr>
      <th>S1003</th>
      <td>23</td>
      <td>Female</td>
      <td>1.0</td>
      <td>3.9</td>
      <td>1.0</td>
      <td>False</td>
      <td>71.0</td>
      <td>9.2</td>
      <td>Poor</td>
      <td>4</td>
      <td>Master</td>
      <td>Good</td>
      <td>1</td>
      <td>True</td>
      <td>26.8</td>
    </tr>
    <tr>
      <th>S1004</th>
      <td>19</td>
      <td>Female</td>
      <td>5.0</td>
      <td>4.4</td>
      <td>0.5</td>
      <td>False</td>
      <td>90.9</td>
      <td>4.9</td>
      <td>Fair</td>
      <td>3</td>
      <td>Master</td>
      <td>Good</td>
      <td>1</td>
      <td>False</td>
      <td>66.4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>S1995</th>
      <td>21</td>
      <td>Female</td>
      <td>2.6</td>
      <td>0.5</td>
      <td>1.6</td>
      <td>False</td>
      <td>77.0</td>
      <td>7.5</td>
      <td>Fair</td>
      <td>2</td>
      <td>High School</td>
      <td>Good</td>
      <td>6</td>
      <td>True</td>
      <td>76.1</td>
    </tr>
    <tr>
      <th>S1996</th>
      <td>17</td>
      <td>Female</td>
      <td>2.9</td>
      <td>1.0</td>
      <td>2.4</td>
      <td>True</td>
      <td>86.0</td>
      <td>6.8</td>
      <td>Poor</td>
      <td>1</td>
      <td>High School</td>
      <td>Average</td>
      <td>6</td>
      <td>True</td>
      <td>65.9</td>
    </tr>
    <tr>
      <th>S1997</th>
      <td>20</td>
      <td>Male</td>
      <td>3.0</td>
      <td>2.6</td>
      <td>1.3</td>
      <td>False</td>
      <td>61.9</td>
      <td>6.5</td>
      <td>Good</td>
      <td>5</td>
      <td>Bachelor</td>
      <td>Good</td>
      <td>9</td>
      <td>True</td>
      <td>64.4</td>
    </tr>
    <tr>
      <th>S1998</th>
      <td>24</td>
      <td>Male</td>
      <td>5.4</td>
      <td>4.1</td>
      <td>1.1</td>
      <td>True</td>
      <td>100.0</td>
      <td>7.6</td>
      <td>Fair</td>
      <td>0</td>
      <td>Bachelor</td>
      <td>Average</td>
      <td>1</td>
      <td>False</td>
      <td>69.7</td>
    </tr>
    <tr>
      <th>S1999</th>
      <td>19</td>
      <td>Female</td>
      <td>4.3</td>
      <td>2.9</td>
      <td>1.9</td>
      <td>False</td>
      <td>89.4</td>
      <td>7.1</td>
      <td>Good</td>
      <td>2</td>
      <td>Bachelor</td>
      <td>Average</td>
      <td>8</td>
      <td>False</td>
      <td>74.9</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 15 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-aafa1f92-e582-49f0-89e3-17c4ae38dd5c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-aafa1f92-e582-49f0-89e3-17c4ae38dd5c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-aafa1f92-e582-49f0-89e3-17c4ae38dd5c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-19d1fa04-5cc8-471f-af05-d2e3658ea47f">
      <button class="colab-df-quickchart" onclick="quickchart('df-19d1fa04-5cc8-471f-af05-d2e3658ea47f')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-19d1fa04-5cc8-471f-af05-d2e3658ea47f button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_153813cd-9b70-4e40-81eb-a7c53b1e08f5">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_153813cd-9b70-4e40-81eb-a7c53b1e08f5 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



## Basic Data Exploration and Summary Statistics
#### (For the tests below, assume 𝛼 = 0.05)

### 1. Chi-Squared Test and Hypothesis Testing

**Question: Does parental education level have an effect on the likelihood of exam score?**

H₀: The parental education level does not have an effect on the likelihood of exam score.

H₁: The parental education level does have an effect on the likelihood of exam score.



```python
con_table = pd.crosstab(df["parental_education_level"], df['exam_score'])
display(con_table)
```



  <div id="df-83f5f751-ea40-4bdd-b139-4ae781c1c9d5" class="colab-df-container">
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
      <th>exam_score</th>
      <th>18.4</th>
      <th>23.1</th>
      <th>26.2</th>
      <th>26.7</th>
      <th>26.8</th>
      <th>27.6</th>
      <th>28.0</th>
      <th>29.5</th>
      <th>29.7</th>
      <th>29.9</th>
      <th>...</th>
      <th>98.3</th>
      <th>98.4</th>
      <th>98.5</th>
      <th>98.7</th>
      <th>98.8</th>
      <th>99.0</th>
      <th>99.3</th>
      <th>99.4</th>
      <th>99.9</th>
      <th>100.0</th>
    </tr>
    <tr>
      <th>parental_education_level</th>
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
      <th>Bachelor</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>19</td>
    </tr>
    <tr>
      <th>High School</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Master</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 453 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-83f5f751-ea40-4bdd-b139-4ae781c1c9d5')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-83f5f751-ea40-4bdd-b139-4ae781c1c9d5 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-83f5f751-ea40-4bdd-b139-4ae781c1c9d5');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-4024ed34-2b59-42eb-aeff-1115e50c911d">
      <button class="colab-df-quickchart" onclick="quickchart('df-4024ed34-2b59-42eb-aeff-1115e50c911d')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-4024ed34-2b59-42eb-aeff-1115e50c911d button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_19d8e588-c4b2-4c49-a4b7-666edb73449b">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('con_table')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_19d8e588-c4b2-4c49-a4b7-666edb73449b button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('con_table');
      }
      })();
    </script>
  </div>

    </div>
  </div>




```python
parent_education = con_table[100.0]
parent_education.plot(kind="bar")
plt.xlabel("Parent Education Level")
plt.ylabel("Exam Score")
plt.title("Parent Education Level vs Exam Score")
```




    Text(0.5, 1.0, 'Parent Education Level vs Exam Score')




    
![png](CMSC320_Checkpoint2_files/CMSC320_Checkpoint2_13_1.png)
    


Across parental education levels, the amount of students earning a perfect exam score is highest among those with at least a bachelor’s degree, but the difference from the high-school group is small. As such, we shouldn’t infer a correlation from this pattern alone, especially since students with parents holding master’s degrees do not show the highest perfect-score rate. Overall, the plot suggests only minor differences across groups and does not provide clear evidence of a meaningful correlation between parental education and earning a 100.


```python
p_value = chi2_contingency(con_table)[1]
display(p_value)
```


    np.float64(0.40411015217143925)


### Conclusion
Because our significance level is 0.05 and the p-value is 0.40411015217143925, we fail to reject the null hypothesis because the p-value is much greater than the significance level. Since we failed to prove the null hypothesis wrong, we can assume that there is no statistically significant evidence to prove the association between the parental education level and their child's exam score.

### 2. Pearson Correlation and Hypothesis Testing

**Question: Do study hours per day have an effect on the likelihood of exam score?**


H₀: Study hours per day do not have an effect on the likelihood of exam_score.

H₁: Study hours per day does have an effect on the likelihood of exam_score.


```python
hours = df['study_hours_per_day']
score = df['exam_score']

r, p_value = pearsonr(hours, score)

print(r)
print(p_value)
plt.scatter(hours, score, color='teal', alpha=0.6, edgecolor='black')

plt.title('Relationship Between Study Hours and Exam Score', fontsize=14)
plt.xlabel('Study Hours per Day', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)

plt.plot(hours, r * (hours - hours.mean()) / hours.std() * score.std() + score.mean(), color='black', linestyle='--', label='Trendline')

plt.show()
```

    0.8254185093960442
    4.595701453345048e-250



    
![png](CMSC320_Checkpoint2_files/CMSC320_Checkpoint2_19_1.png)
    


### Conclusion
We reject the null hypothesis because the p-value is an extremely small number close to 0, which is way smaller than the level of significance, alpha = 0.05. This indicates the observed data is statistically significant and that there is a very small likelihood of falsely rejecting the null. So I am confident that there could be a relation between the number of hours studied per day and the exam score.


```python
sns.set(style="whitegrid")
sns.regplot(x='study_hours_per_day', y='exam_score', data=df)
plt.title('Study Hours vs Exam Score')
plt.xlabel('Study Hours per Day')
plt.ylabel('Exam Score')
plt.show()
```


    
![png](CMSC320_Checkpoint2_files/CMSC320_Checkpoint2_21_0.png)
    


### 3. T-Test and Hypothesis Testing

**Question: Does gender have an effect on the likelihood of the number of study hours?**


H₀: Gender does not have an effect on the likelihood of the number of study hours.

H₁: Gender does have an effect on the likelihood of the number of study hours.


```python
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

female = df[df['gender']== 'Female']['study_hours_per_day']
male = df[df['gender']== 'Male']['study_hours_per_day']

results= ttest_ind(female, male)

print(results.pvalue)

plt.boxplot([female, male], tick_labels=['Female', 'Male'], patch_artist=True)

plt.title('Relationship of Study Hours per Day and Gender')
plt.ylabel('Study Hours per Day')

plt.show()
```

    0.446687018132807



    
![png](CMSC320_Checkpoint2_files/CMSC320_Checkpoint2_24_1.png)
    


### Conclusion
We established that the significance level is 0.05. From the t-test, we calculated a p-value of 0.47. Therefore, by comparing the two values we can conclude that p > a. Because p > a, we fail to reject the null hypothesis. This means we can assume that there is no statistically significant evidence to prove the association between gender and study hours per day.


```python
!jupyter nbconvert --to markdown CMSC320_Checkpoint2.ipynb --embed-images
```

    [NbConvertApp] Converting notebook CMSC320_Checkpoint2.ipynb to markdown
    [NbConvertApp] Support files will be in CMSC320_Checkpoint2_files/
    [NbConvertApp] Making directory CMSC320_Checkpoint2_files
    [NbConvertApp] Writing 58944 bytes to CMSC320_Checkpoint2.md


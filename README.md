## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd 
df=pd.read_csv("Encoding Data.csv") 
df
```
<img width="352" height="377" alt="image" src="https://github.com/user-attachments/assets/564a374f-d31b-4037-9bd5-83e8a040629a" />
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="240" height="195" alt="image" src="https://github.com/user-attachments/assets/1d329197-407b-429b-b25b-5406a5c3b7c3" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]]) 
df
```
<img width="367" height="377" alt="image" src="https://github.com/user-attachments/assets/5c18260f-67e7-4940-bb93-f30b7d5d7867" />

```
le=LabelEncoder() 
dfc=df.copy() 
dfc['ord_2']=le.fit_transform(dfc['ord_2']) 
dfc
```
<img width="392" height="378" alt="image" src="https://github.com/user-attachments/assets/2976d0e3-6637-4423-8312-6951a03dc785" />
```
from sklearn.preprocessing import OneHotEncoder 
ohe=OneHotEncoder(sparse_output=False) 
df2=df.copy() 
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) 
df2=pd.concat([df2,enc],axis=1) 
df2
```
<img width="545" height="370" alt="image" src="https://github.com/user-attachments/assets/a69f2b19-c2c5-4404-b98d-2653426c8940" />
```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="662" height="392" alt="image" src="https://github.com/user-attachments/assets/60028ba7-6f8b-4a83-9d46-28f937d11388" />
```
from category_encoders import BinaryEncoder 
df=pd.read_csv("data.csv") 
df
```
<img width="471" height="385" alt="image" src="https://github.com/user-attachments/assets/da0928ed-6922-4a21-8d53-6718359f3479" />
```
be=BinaryEncoder() 
nd=be.fit_transform(df['Ord_2']) 
dfb=pd.concat([df,nd],axis=1) 
dfb
```
<img width="687" height="395" alt="image" src="https://github.com/user-attachments/assets/4a547099-3916-4ade-afaa-442e7cb4b1db" />
```
from category_encoders import TargetEncoder 
te=TargetEncoder() 
CC=df.copy() 
new=te.fit_transform(X=CC["City"],y=CC["Target"]) 
CC=pd.concat([CC,new],axis=1) 
CC
```
<img width="562" height="375" alt="image" src="https://github.com/user-attachments/assets/cb05a555-1de0-46b0-83ef-68c8657fe6a3" />
```
import pandas as pd 
from scipy import stats 
import numpy as np 
df=pd.read_csv("Data_to_Transform.csv") 
df
```
<img width="812" height="453" alt="image" src="https://github.com/user-attachments/assets/6041b22b-94aa-49da-8a21-6020c693fe4e" />
```
df.skew()
```
<img width="291" height="92" alt="image" src="https://github.com/user-attachments/assets/b456f493-6d42-4ded-a750-afc059119092" />
```
np.log(df["Highly Positive Skew"])
```
<img width="513" height="241" alt="image" src="https://github.com/user-attachments/assets/1ab45316-e8fe-477c-a7a8-6a1449099ca6" />
```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="558" height="232" alt="image" src="https://github.com/user-attachments/assets/0a77693a-1435-4820-af35-96df68cba9de" 
```
np.sqrt(df["Highly Positive Skew"])
```
<img width="483" height="232" alt="image" src="https://github.com/user-attachments/assets/c4dc3333-281a-4b44-808a-c6f86adb7d5d" />

```
np.square(df["Highly Positive Skew"])
```
<img width="517" height="230" alt="image" src="https://github.com/user-attachments/assets/21d8daa1-d563-4490-8b27-d32e3f3402cc" />
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
df
```
<img width="832" height="472" alt="image" src="https://github.com/user-attachments/assets/012cb799-47dd-4d76-892f-33954b9c7b84" />
```
df.skew()
```
<img width="367" height="122" alt="image" src="https://github.com/user-attachments/assets/fbfdd865-53eb-4412-a9cc-c90c37e211c1" />
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
df.skew()
```
<img width="432" height="145" alt="image" src="https://github.com/user-attachments/assets/9f216f6a-db78-416b-b362-86c04d8dd22d" />
```
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal') 
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
df
```
<img width="801" height="481" alt="image" src="https://github.com/user-attachments/assets/f3de78b3-1fab-4257-b0f5-be489d398e24" />
```
import seaborn as sns 
import statsmodels.api as sm 
import matplotlib.pyplot as plt 
sm.qqplot(df["Moderate Negative Skew"],line='45')  
plt.show()
```
<img width="762" height="491" alt="image" src="https://github.com/user-attachments/assets/82b34b7a-416e-479b-9ce0-edb3b591e7e0" />
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL 
plt.show()
```
<img width="687" height="486" alt="image" src="https://github.com/user-attachments/assets/e259cc78-a578-417d-8f12-0c59966a5706" />

```
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891) 
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()
```
<img width="790" height="483" alt="image" src="https://github.com/user-attachments/assets/c0274d4a-e041-452d-8daa-4350e0540bc3" />










  



















     
# RESULT:
       # INCLUDE YOUR RESULT HERE

       

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
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="253" height="336" alt="image" src="https://github.com/user-attachments/assets/ec4dfaba-8941-415a-936d-2e159f918fb2" />
```
 from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
 pm=['Hot','Warm','Cold']
 e1=OrdinalEncoder(categories=[pm])
 e1.fit_transform(df[["ord_2"]])
```
<img width="113" height="172" alt="image" src="https://github.com/user-attachments/assets/aee28c7b-110f-4cae-a2a4-1b34b6836dc7" />
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="277" height="333" alt="image" src="https://github.com/user-attachments/assets/b7c3201d-9406-4d30-8b34-897e45282924" />
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="292" height="336" alt="image" src="https://github.com/user-attachments/assets/331637ef-f097-4d93-bbf6-4a90a170d46d" />
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) # Orders in Alphabetical Order Blue , Green, Red
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="512" height="456" alt="image" src="https://github.com/user-attachments/assets/2c78bcaf-76c8-485e-b7e9-8815bd64cef2" />
```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="509" height="443" alt="image" src="https://github.com/user-attachments/assets/fc7e5229-f479-451d-9d4a-5c624e112cfe" />
```
pip install --upgrade category_encoders
```
<img width="1138" height="316" alt="image" src="https://github.com/user-attachments/assets/b285cf9a-eddd-4f8c-8b35-7e66b5ae6082" />
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
<img width="585" height="456" alt="image" src="https://github.com/user-attachments/assets/4198d7d0-49f9-48c9-8afc-a95e3d467c28" />
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="837" height="450" alt="image" src="https://github.com/user-attachments/assets/f29d0666-dd73-4e49-b1ac-5d73587f33cb" />
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="848" height="460" alt="image" src="https://github.com/user-attachments/assets/6a4c98b1-3c03-4727-856a-98e831d55919" />
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="712" height="380" alt="image" src="https://github.com/user-attachments/assets/58eb9916-0e40-433f-8f9b-15e08472bb61" />
```
df.skew()
```
<img width="354" height="265" alt="image" src="https://github.com/user-attachments/assets/7cb84165-b9a4-4e53-8f25-0e300974bb11" />
```
np.log(df["Highly Positive Skew"])
```
<img width="311" height="524" alt="image" src="https://github.com/user-attachments/assets/13e4128a-fc55-4ca6-98a8-c8c9c7b1e05d" />
```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="337" height="531" alt="image" src="https://github.com/user-attachments/assets/aca32f31-388f-4bab-811e-452be442a79f" />
```
np.sqrt(df["Highly Positive Skew"])
```
<img width="325" height="522" alt="image" src="https://github.com/user-attachments/assets/0702f892-94d8-4c0c-9d1c-71f5c470d0da" />
```
np.square(df["Highly Positive Skew"])
```
<img width="220" height="414" alt="image" src="https://github.com/user-attachments/assets/6281a255-fef3-4224-9fc9-a0e807e5de43" />
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="921" height="399" alt="image" src="https://github.com/user-attachments/assets/f44c5e8b-2ed9-4b63-bc35-788a1fd09b45" />
```
df.skew()
```
<img width="399" height="248" alt="image" src="https://github.com/user-attachments/assets/e52c6379-0e90-4a9a-acdf-f703142f79ac" />
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="466" height="312" alt="image" src="https://github.com/user-attachments/assets/222ef017-7804-4a75-9f5a-ad7ed8f46383" />
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1103" height="392" alt="image" src="https://github.com/user-attachments/assets/bc34b330-f235-41b4-8819-0fe2223126e7" />
```
import seaborn as sns
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT
plt.show()
```
<img width="755" height="573" alt="image" src="https://github.com/user-attachments/assets/04abb8a6-75f6-4c6f-96d3-420a0979ef97" />
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL
plt.show()
```
<img width="735" height="561" alt="image" src="https://github.com/user-attachments/assets/33ea39b6-cc63-468c-9e84-1ca837bca5c8" />
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="720" height="568" alt="image" src="https://github.com/user-attachments/assets/fa5ae9fd-d701-49b9-b438-dc5d592fbe01" />

# RESULT:
      
Thus the commands are executed successfully.
       

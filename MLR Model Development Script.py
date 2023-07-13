# Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import sklearn.model_selection as model_selection
import pickle

# Read the data from the Excel file
df = pd.read_excel(r"C:\Users\research.modeling\Desktop\Prediction Models\HSM Property Prediction Model\HSM 1\datasets\HSM-1.xlsx","model")

print("##################### Generating XGboost Model #####################")

# Initialize Features
X = df.drop(['COIL_ID', 'GRADE', 'COIL_GEN_TIME', 'UTS', 'YS', 'P_ELONGATION', 'UTS_PRED', 'YS_PRED', 'EL_PRED'], axis = 1)# Initialize Target

Y1 = df['UTS']
Y2 = df['YS']
Y3 = df['P_ELONGATION']

ts=0.3
rs1=0
rs2=0

X1_train, X1_test, Y1_train, Y1_test = model_selection.train_test_split(X,Y1,test_size=ts,random_state=rs1)
X2_train, X2_test, Y2_train, Y2_test = model_selection.train_test_split(X,Y2,test_size=ts,random_state=rs1)
X3_train, X3_test, Y3_train, Y3_test = model_selection.train_test_split(X,Y3,test_size=ts,random_state=rs2)

# Do linear fit
lr1 = LinearRegression()
lr1.fit(X1_train, Y1_train)             # Fit the model
pred1 = lr1.predict(X1_test)
pickle.dump(lr1, open(r'C:\Users\research.modeling\Desktop\Prediction Models\HSM Property Prediction Model\HSM 1\models\UTS_MLR.pkl', 'wb'))
print('UTS R2:', np.sqrt(metrics.r2_score(Y1_test, pred1)))

lr2 = LinearRegression()
lr2.fit(X2_train, Y2_train)             # Fit the model
pred2 = lr2.predict(X2_test)
pickle.dump(lr2, open(r'C:\Users\research.modeling\Desktop\Prediction Models\HSM Property Prediction Model\HSM 1\models\YS_MLR.pkl', 'wb'))
print('YS R2:', np.sqrt(metrics.r2_score(Y2_test, pred2)))

lr3 = LinearRegression()
lr3.fit(X3_train, Y3_train)             # Fit the model
pred3 = lr3.predict(X3_test)
pickle.dump(lr3, open(r'C:\Users\research.modeling\Desktop\Prediction Models\HSM Property Prediction Model\HSM 1\models\EL_MLR.pkl', 'wb'))
print('EL R2:', np.sqrt(metrics.r2_score(Y3_test, pred3)))

print("MLR model generated !!!")

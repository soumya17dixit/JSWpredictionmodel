"""
Created on Sat Sep 18 23:15:57 2020

Author: Pranav Kumar Tripathi (R&D and SS)
"""
#from sklearn.linear_model import LinearRegression
import pandas as pd
#import numpy as np
import joblib
import cx_Oracle


######## Establish DB Connection ########
dsn_tns = cx_Oracle.makedsn('vip-jswvdbp60.jsw.in', '1525', service_name='SPTSP.jsw.in') 
db_oracle = cx_Oracle.connect(user=r'HTS_ANALYTICS', password='predana', dsn=dsn_tns)
dbcursor = db_oracle.cursor()


######## Load Prediction Models ########
loaded_model_UTS = joblib.load(r"C:\Users\research.modeling\Desktop\Prediction Models\HSM Property Prediction Model\HSM 1\models\UTS_MLR.pkl")
loaded_model_YS = joblib.load(r"C:\Users\research.modeling\Desktop\Prediction Models\HSM Property Prediction Model\HSM 1\models\YS_MLR.pkl")
loaded_model_EL = joblib.load(r"C:\Users\research.modeling\Desktop\Prediction Models\HSM Property Prediction Model\HSM 1\models\EL_MLR.pkl")

##loaded_model_UTS = joblib.load(r"C:\Users\research.modeling\Desktop\Prediction Models\HSM Property Prediction Model\HSM 1\models\UTS_MLR.pkl")
##loaded_model_YS = joblib.load(r"C:\Users\research.modeling\Desktop\Prediction Models\HSM Property Prediction Model\HSM 1\models\YS_MLR.pkl")
##loaded_model_EL = joblib.load(r"C:\Users\research.modeling\Desktop\Prediction Models\HSM Property Prediction Model\HSM 1\models\EL_MLR.pkl")

######## Read Data ########

hsm_data_read_query = "SELECT COIL_ID, GRADE, FURN_EXIT_TEMP, FM_DOUT_TEMP, COILING_TEMP, SLAB_WID, SLAB_LEN, STRIP_EXIT_LEN, STRIP_EXIT_THIK, FM_EXIT_THIK, AL, B, C, CA, NB, CR, CU, MN, MO, N, NI, P, S, SI, TI, V from HSM_YS_UTS_PRED_ENTB WHERE RM_DOUT_TEMP IS NULL AND FURN_EXIT_TEMP>500 AND FM_DOUT_TEMP>500 AND COILING_TEMP>250 AND FM_EXIT_THIK>0.5 AND COIL_ID>23000000 AND COIL_ID<23500000 AND COIL_GEN_TIME>=SYSDATE - 5 AND COIL_RED_STS='N' ORDER BY COIL_ID ASC"
dbcursor.execute(hsm_data_read_query)
coil_list_with_parameters = dbcursor.fetchall()     # gives list of tuples
#print(coil_list_with_parameters)


no_of_coils=len(coil_list_with_parameters)

if(no_of_coils)>0:

    print(str(no_of_coils) + " coils retrieved from production database. Prediction will be done.")

    print("--------------------------------------------------------------------")

    df=pd.DataFrame(coil_list_with_parameters)              # converting database data to dataframe
    col_names = [row[0] for row in dbcursor.description]    # fetching column names
    df.columns=col_names                                    # adding column names

#    print(len(df.index))

    df_filtered=df.dropna()                                 # dropping rows with NULL values
    df_filtered.reset_index(inplace=True)                   # resetting index 
    df_filtered=df_filtered.drop(['index'], axis=1)

#    print(df_filtered.columns)
  
    df_predict=df_filtered.drop(['COIL_ID','GRADE'], axis=1)

    try:
        ######## Prediction using loaded models ########

        predicted_uts=loaded_model_UTS.predict(df_predict)
        predicted_ys=loaded_model_YS.predict(df_predict)
        predicted_el=loaded_model_EL.predict(df_predict)

        print("The predicted UTS : ",predicted_uts)
        print("The predicted YS : ",predicted_ys)
        print("The predicted %EL : ",predicted_el)

        
        df_UTS = pd.DataFrame(predicted_uts)
        df_YS = pd.DataFrame(predicted_ys)
        df_EL = pd.DataFrame(predicted_el)

        df_results=pd.concat([df_UTS, df_YS, df_EL], axis=1, sort=False)

        #print(df_results)

        df_results.columns=['PRED_UTS', 'PRED_YS', 'PRED_EL']

        df_results = df_results.round(decimals=2)


        ######## Save Full Data with prediction results to Excel File ########
        df_full_data_with_results=pd.concat([df_filtered,df_results], axis=1, sort=False)

##        df_full_data_with_results.to_excel(r"C:\Users\research.modeling\Desktop\Prediction Models\HSM Property Prediction Model\HSM 1\prediction\Prediction Results.xlsx", index = False, header=True)
##
##        print("Excel File Created")

        ######## Save Data into DB from dataframe ########

        for ind in df_full_data_with_results.index: 
            coil_id=df_full_data_with_results['COIL_ID'][ind]

            hsm_data_write_uts_query = "UPDATE HSM_YS_UTS_PRED_ENTB SET UTS_PRED = '" + str(df_full_data_with_results['PRED_UTS'][ind]) + "' WHERE COIL_ID = " + coil_id
            hsm_data_write_ys_query = "UPDATE HSM_YS_UTS_PRED_ENTB SET YS_PRED = '" + str(df_full_data_with_results['PRED_YS'][ind]) + "' WHERE COIL_ID = " + coil_id
            hsm_data_write_el_query = "UPDATE HSM_YS_UTS_PRED_ENTB SET EL_PRED = '" + str(df_full_data_with_results['PRED_EL'][ind]) + "' WHERE COIL_ID = " + coil_id
            coil_read_status_query = "UPDATE HSM_YS_UTS_PRED_ENTB SET COIL_RED_STS = 'Y' WHERE COIL_ID = " + coil_id
            prediction_time_query = "UPDATE HSM_YS_UTS_PRED_ENTB SET PRED_TIME = SYSDATE WHERE COIL_ID = " + coil_id

            dbcursor.execute(hsm_data_write_uts_query)
            dbcursor.execute(hsm_data_write_ys_query)
            dbcursor.execute(hsm_data_write_el_query)
            dbcursor.execute(coil_read_status_query) 
            dbcursor.execute(prediction_time_query)    

            db_oracle.commit()

##            print("UTS, YS & %EL predicted and inserted into database")   

    except:

        print("Some error occurred...probably missing data or No new coil retrieved, execution will continue!!!")
else:

    print("No coils retrieved. Script will relook for coils again.")

print("All Done")


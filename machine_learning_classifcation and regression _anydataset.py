
import streamlit as st 
import pandas as pd 

import os 
import pickle
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE
from scipy.stats import yeojohnson
from scipy.stats import shapiro 
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import OneHotEncoder,StandardScaler ,MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import skew
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek
@st.cache_data(show_spinner=False)
def load_data(files) :
  return pd.read_csv(files)

UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


files = st.file_uploader("Upload CSV", accept_multiple_files=False)

if files is not None:
    df = load_data(files)  
    save_path = os.path.join(UPLOAD_FOLDER, "latest_upload.csv")
    df.to_csv(save_path, index=False)
    
    st.success(f"Data saved to {save_path}")
    st.dataframe(df) 
    n_rows = st.number_input(
        "Number of rows to display",
        min_value=5,
        max_value=len(df),
        value=5,
        step=1
    )
    column_to_show = st.multiselect("Select columns to display", options=df.columns.tolist(), default=df.columns.tolist())
   
    st.dataframe(df[:n_rows][column_to_show])
    col1 , col2 , col3 =st.columns(3)
    tab1 , tab2 = st.tabs(["Scatter Plot", "Histogram"] )
    with tab1 : 
       num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
       with col1 :
           x_axis = st.selectbox("Select X-axis", options=num_cols)   
       with col2 :
           y_axis = st.selectbox("Select Y-axis", options=num_cols)     
       with col3  : 
          color = st.selectbox("Select Color", options=num_cols)
       
       fig_scatter = px.scatter(df, x=x_axis, y=y_axis,title=f"Scatter plot of {x_axis} vs {y_axis}", color = color )
       st.plotly_chart(fig_scatter)
    with tab2:
      feature_select = st.selectbox("Select feature for histogram", options=num_cols)
      fig_hist = px.histogram(df, x=feature_select, title=f"Histogram of {feature_select}", color=color)  
      st.plotly_chart(fig_hist)  


    options = ["-- select target --"] + df.columns.tolist()
    target = st.selectbox("Select target column", options=options)

    if target == "-- select target --":
      st.warning("Please select a target column to continue.")
      st.stop()


    st.success(f"Selected target column: {target}")

    t1 , t2 = st.tabs(["regression", "classification"])
    def clamp_outliers(df, target=None):
     numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

     if target in numeric_cols:
        numeric_cols.remove(target)

  
     df_clamped = df.copy()

     for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clamped[col] = np.where(
            df[col] < lower_bound, lower_bound,
            np.where(df[col] > upper_bound, upper_bound, df[col])
        )

     return df_clamped
    # function to drop column if contain redundant values is more than 80% from columns
    def drop_redundant_columns(df, threshold=0.8):
      columns_to_drop = []
      for col in df.columns:
         most_freq_ratio = df[col].value_counts(normalize=True).max()
         if most_freq_ratio > threshold:
            columns_to_drop.append(col)
      return columns_to_drop
  
    columns = drop_redundant_columns(df, threshold=0.8)
    xfound = True
    if target in columns:
       st.write(f"Target column '{target}' is redundant and will be removed.")
       columns.remove(target)
       xfound = False
    else : 
       st.write(f"Target column '{target}' is not redundant.")
    df.drop(columns=columns, inplace= True )
    
    df = clamp_outliers(df)
    def null_percentage(df):
       null_percent = (df.isnull().sum() /df.shape[0]) * 100
       return null_percent.reset_index().rename(columns={'index':'column', 0:'null_percentage'})

      
    df.drop(columns = null_percentage(df)[null_percentage(df)['null_percentage']>=80]['column'].tolist()   , inplace = True  )
     
    columns = null_percentage(df)[
    (null_percentage(df)['null_percentage']  > 0 ) &
    (null_percentage(df)['null_percentage'] < 80)
     ]['column'].tolist()
    

    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target in num_cols and df[target].nunique() < 10:
       num_cols.remove(target)
    for i in columns :
       if i in num_cols:
           df[i].fillna(0, inplace=True)
       else :
           df[i].fillna(df[i].mode()[0], inplace=True)
    
 
# correlation
# drop feature is high correlation with another dependent feature and low with independant feature
    def drop_high_correlation_features(df, target, threshold=0.8):
        corr_matrix = df[num_cols].corr()
        to_drop = set() 
        features = list(corr_matrix.columns)

        for i in range(len(features)):
           for j in range(i + 1, len(features)):
             col1 = features[i]
             col2 = features[j]
             
             if abs(corr_matrix.loc[col1, col2]) > threshold:
            
                if abs(corr_matrix.loc[col1, target]) < abs(corr_matrix.loc[col2, target]):
                    to_drop.add(col1)
                else:
                    to_drop.add(col2)

        return list(to_drop)
    if target in num_cols : 
      columns = drop_high_correlation_features(df, target=target, threshold=0.8)
      if len(df.columns) > 5 and target not in columns:
        df = df.drop(columns=columns)
      elif len(df.columns) > 5 and target in columns:
       columns.remove(target)
       df = df.drop(columns=columns)

    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    found = True
    def fixed_skewness(df, skew_threshold=0.5):
       """Fix skewed numerical features."""
       skewed_feats = df[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
       skewness = pd.DataFrame({'Skew': skewed_feats})
       skewness = skewness[abs(skewness) > skew_threshold]
       for col in skewness.index:
          if col == target : 
            found = False
          else  : 
            min_val = df[col].min()
            if min_val <= -1:
              df[col] = df[col] + abs(min_val) + 1  # إزاحة القيم كلها لتكون ≥ 0
            df[col] = np.log1p(df[col])
       return df
    df = fixed_skewness(df)
    std_feature = []
    min_max_feature =[]
    for col in num_cols : 
      stats , p_val = shapiro(df[col])

    print (f"the stats to {col} is {stats} and the p-value is {p_val}  ")
    if p_val < 0.05:
      min_max_feature.append(col)
    else :
      std_feature.append(col)

    std = StandardScaler() 
    if target in min_max_feature :
      min_max_feature.remove(target)     
    elif target in std_feature :
      std_feature.remove(target) 
    for col in std_feature :
     if len(std_feature)!=0 and col != target:
       df[std_feature] = std.fit_transform(df[std_feature])
    for col in min_max_feature :
      if len(min_max_feature)!=0 and col != target :
        min_max = MinMaxScaler()
        df[min_max_feature] = min_max.fit_transform(df[min_max_feature])
    
    def split_features_by_encoding(df, categorical_cols, ordinal_mappings=None):
      onehot_features = []
      label_features = []
      binary_features = []
      ordinal_features = []

      for col in categorical_cols:
        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)

       
        if ordinal_mappings and col in ordinal_mappings:
            ordinal_features.append(col)

        elif n_unique == 2:
            binary_features.append(col)

        elif 2 < n_unique <= 10:
            onehot_features.append(col)

        elif 10 < n_unique <= 50:
            label_features.append(col)
        else:
             ordinal_features.append(col)
      return onehot_features, label_features, binary_features, ordinal_features
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    onehot_features, label_features, binary_features, ordinal_features = split_features_by_encoding (df ,categorical_cols=cat_cols, ordinal_mappings=None)
    if len(onehot_features) !=0 :
       encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 
         
       df_encoded= encoder.fit_transform(df[onehot_features])   
       encoded_cat_df = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out() , index=df.index)
       df.drop(columns=onehot_features, inplace=True)  
       df = pd.concat([df, encoded_cat_df], axis=1)     
   
    if len(label_features) != 0:
       lbe = LabelEncoder()
       df[label_features] = df[label_features].apply(lbe.fit_transform)
       
    if len(ordinal_features) != 0:
       oe = OrdinalEncoder()
       df[ordinal_features] = oe.fit_transform(df[ordinal_features])
    if len(binary_features) != 0: 
      be = LabelEncoder()
      binary_features_existing = [col for col in binary_features if col in df.columns]
      for col in binary_features_existing:
         df[col] = be.fit_transform(df[col])
    df.fillna(0 , inplace = True )
   
    x = df.drop(columns=[target])
    y = df[target] 
    if xfound == False  : 
      x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42 , stratify=y )
    else:
      x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42 )
    
    if xfound == False :
       from imblearn.combine import SMOTETomek
       from imblearn.over_sampling import SMOTE
       smt = SMOTETomek(
    random_state=42, 
    smote=SMOTE(k_neighbors=1, random_state=42)
)
       x_resample, y_resample = smt.fit_resample(x_train, y_train)

    with t1:   
        x_train = x_train.dropna()
        y_train = y_train.loc[x_train.index]
        reg_model = st.selectbox("Select regression model ", options=['linear regression' , 'xgboostregressor'], key='reg_model_tab1') 
          
        if reg_model == 'linear regression':
          lin_reg = LinearRegression()
          lin_reg.fit(x_train, y_train)
          if found ==False : 
            y_test = np.expm1(y_test)
          y_pred = lin_reg.predict(x_test)
          mse_lr = mean_squared_error(y_test, y_pred)
          rmse_lr = np.sqrt(mse_lr)
          r2 = r2_score(y_test, y_pred)
          print(f"Linear Regression RMSE: {rmse_lr}, MSE: {mse_lr}, R2: {r2}")
          st.write("### Linear Regression")
          st.write(f"RMSE: {rmse_lr}")
          st.write(f"MSE: {mse_lr}")
          st.write(f"R2: {r2}")
        elif reg_model == 'xgboostregressor':
          xgb_model = XGBRegressor(
        n_estimators=150,

        learning_rate=.02,
        max_depth=5,
        random_state=0
      
        )
          x_train = x_train.dropna()
          y_train = y_train.loc[x_train.index]
          xgb_model.fit(x_train, y_train)
          if found ==False : 
            y_test = np.expm1(y_test)
          y_pred = xgb_model.predict(x_test)
          mse_xgb = mean_squared_error(y_test, y_pred)
          rmse_xgb = np.sqrt(mse_xgb)
          r2_xgb = r2_score(y_test, y_pred)
          print(f"XGBoost RMSE: {rmse_xgb}, MSE: {mse_xgb}, R2: {r2_xgb}")

          st.write("###   XGBoost")
          st.write(f"RMSE: {rmse_xgb}")
          st.write(f"MSE: {mse_xgb}")
          st.write(f"R2: {r2_xgb}")

    with t2:

        class_model  = st.selectbox("Select classification model", options=["-- Select --" , 'knn' , 'logistic_regression', 'xgb_classifier' ,'svm'], key="class_model_tab2"  ) 
        if xfound == True  : 
         if class_model =="knn" : 
            knn_model = KNeighborsClassifier(n_neighbors=5)
            if found ==False : 
              y_train = np.expm1(y_train)
            x_train = x_train.dropna()
            y_train = y_train.loc[x_train.index]
            knn_model.fit(x_train, y_train)
            y_pred = knn_model.predict(x_test)
            st.write("### KNN Metrics")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
            st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
            st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
            st.write(f"F1: {f1_score(y_test, y_pred, average='weighted')}")

         elif class_model == "xgb_classifier":
           xgb_model = XGBClassifier(
             n_estimators=100,
               learning_rate=0.05,
             max_depth=5,
             random_state=0,
           use_label_encoder=False,
             eval_metric="logloss"
            )
           if found ==False : 
              y_train = np.expm1(y_train)
           x_train = x_train.dropna()
           y_train = y_train.loc[x_train.index]
           xgb_model.fit(x_train, y_train)
           y_pred = xgb_model.predict(x_test)
           st.write("### XGBoost Classifier")
           st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
           st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
           st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
           st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")

         elif class_model == "logistic_regression":
           log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
           if found ==False : 
              y_train = np.expm1(y_train)
           x_train = x_train.dropna()
           y_train = y_train.loc[x_train.index]
           log_reg_model.fit(x_train, y_train)
           y_pred = log_reg_model.predict(x_test)
           st.write("### Logistic Regression")
           st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
           st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
           st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
           st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
        
         elif class_model == "svm":
          svm_model = SVC(kernel='poly' , degree = 3 , verbose=True   , random_state=42, C=1, gamma='scale')
          if found ==False : 
              y_train = np.expm1(y_train)
          x_train = x_train.dropna()
          y_train = y_train.loc[x_train.index] 
          svm_model.fit(x_train, y_train)
          y_pred = svm_model.predict(x_test)
          st.write("### Support Vector Machine (SVM)")
          st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
          st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
          st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
          st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
        else: 
         if class_model =="knn" : 
            knn_model = KNeighborsClassifier(n_neighbors=5)
            if found ==False : 
              y_train = np.expm1(y_train)
            knn_model.fit(x_resample, y_resample)
            y_pred = knn_model.predict(x_test)
            st.write("### KNN Metrics")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
            st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
            st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
            st.write(f"F1: {f1_score(y_test, y_pred, average='weighted')}")

         elif class_model == "xgb_classifier":
           xgb_model = XGBClassifier(
             n_estimators=100,
               learning_rate=0.05,
             max_depth=5,
             random_state=0,
           use_label_encoder=False,
             eval_metric="logloss"
            )
           if found ==False : 
              y_train = np.expm1(y_train)
           xgb_model.fit(x_resample, y_resample)
           y_pred = xgb_model.predict(x_test)
           st.write("### XGBoost Classifier")
           st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
           st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
           st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
           st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")

         elif class_model == "logistic_regression":
           log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
           if found ==False : 
              y_train = np.expm1(y_train)
           log_reg_model.fit(x_resample, y_resample)
           y_pred = log_reg_model.predict(x_test)
           st.write("### Logistic Regression")
           st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
           st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
           st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
           st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
        
         elif class_model == "svm":
          svm_model = SVC(kernel='rbf', random_state=42, sigma=0.02, C=.5 , gamma='scale' )

          if found ==False : 
              y_train = np.expm1(y_train)
          svm_model.fit(x_resample, y_resample)
          y_pred = svm_model.predict(x_test)
          st.write("### Support Vector Machine (SVM)")
          st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
          st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
          st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
          st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
           
           

else:
    st.warning("Please upload a CSV file to continue.")




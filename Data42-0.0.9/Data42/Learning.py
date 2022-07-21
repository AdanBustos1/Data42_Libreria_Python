import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import FeatureHasher

def generar_muestras_balanceo(dataset, columna):
    """ Esta función permite crear muestras sintéticas, a partir de conjuntos de datos de clasificación desequilibrados.
        Para emplear esta función es necesario (import : Numpy, pandas, from imblearn.over_sampling import SMOTE)
        Se ayuda de la función SMOTE (Tecnica de sobremuestreo de minorias sintéticas)
        Se basa en la creación de muestras sinteticas, en el grupo/grupos minoritario(s).
    Parameters:
            -dataset: Set de datos desbalanceado
            -columna: Tiene que estar incluída en el set de datos desbalanceado, puede ser de tipo "string" ó "númerico". 
                      Esta columna representara a la clase o grupo que pertence mi muestra (ejemplo: Tipo, Clase, Grupo, Ypredict)
    
    Output: Regresa el dataset balanceado (el cual contiene las muestras originales, como las muestras creadas)
    """
    X= dataset.drop([columna],1)
    y= dataset[columna]

    smt= SMOTE(random_state=42)
    X_train_sm, y_train_sm = smt.fit_resample(X, y)

    dataset_balanceado= pd.concat([X_train_sm,y_train_sm], axis=1)
    
    return dataset_balanceado


def balanceado(df, target):
    list_b = []

    for index,value in df[target].value_counts().items():
        list_b.append({'Categoria': index, 'Peso': round(float(value )/ len(df[target])*100, 2)})

    list_b = pd.DataFrame(list_b)

    return list_b
    

def correlation_coeff(dataset,threshold):
    ''' 
        The correlation coefficient is a statistical measure of the strength of the relationship 
        between the relative movements of two variables.
        Parameters X_train ,Threshold where threshold is 
        the maximimum percent in correlation in the independants variables will be high correlated.
        X_train is the dataset, in this case we just interestd on the indepedant features, that wy we use X_train.
        threshold values must bue from 0 to 1 like, 0.5, 0.7, 0.9.
        Function will return witch variable have a high correlation in indepeant features.
    '''
    col_corr = set() #Set of all the names of corralated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i] #guetting the name of column
                col_corr.add(colname)
    return col_corr


def balance_observation(df, target):
    '''
    Function to know the balance classes.
    Input:
        - df: dataframe
        - target: column
    Output: 
        Dataframe with the categories of the column and their weight.  

    Libraries used:
        import pandas as pd
        import numpy as np    
    '''

    new_list = []
    for index, value in df[target].value_counts().items():
        new_list.append({'Category': index, 'Weight': round(float(value )/ len(df[target])*100, 2)})

    return pd.DataFrame(new_list)


def replace_missings_train_test(X_test, X_train, col):
    '''
	Function to eliminate or replace missings. This version is for dataframes that have been separated into train and test.
	The user is asked for the dataframe and the column that they want to act on, and they are given options to choose what they want to
	do with the missings.

	Parameters:
	                - df: dataframe
	                - col: column you want to modify

	Libraries used:
	    import pandas as pd
	    import numpy as pd                
	'''

    tipo = str(X_test[col].dtype)
    if 'int' in tipo or 'float' in tipo:
        op = str(input("Enter Mean to change your missings to the mean or Median to change them to the median"))
        if op.lower() == "mean":
            X_train[col] = X_train[col].fillna(X_train[col].mean())
            X_test[col]  = X_test[col].fillna(X_train[col].mean())
            print("The missings of your column have been replaced by the mean of train on train and test sets")
        elif op.lower() == "median":
            X_train[col] = X_train[col].fillna(X_train[col].median())
            X_test[col]  = X_test[col].fillna(X_train[col].median())
            print("The missings of your column have been replaced by the median of train on train and test sets")
    else:
        print("Your variable is not numerical, changing the missings in train and test sets by the mode of train...")
        X_train[col] = X_train[col].fillna(X_train[col].mode().iloc[0])
        X_test[col]  = X_test[col].fillna(X_train[col].mode().iloc[0])
        print("The missings of your column have been replaced in train and test sets by the mode of train")


def models():
    """
    Function which summarizes the different machine learning models available for supervised learning
    Inputs:
    
    Output:
        Description of the different models suitable for classifcation or regression problems.
    """


    text= """Below you will find a list with the available machine learning models for supervised learning included in this library.

    According to the desired results of your model (regression or classification) you may choose among the following:

        Classification

            - Logitic regression: Logistic regression is a statistical analysis method to predict a binary outcome, such as yes or no, based on prior observations of a data set.
              
              A logistic regression model predicts a dependent data variable by analyzing the relationship between one or more existing independent variables.
            
            - SVM: Or Suport Vector Machine,  the objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.

            - Decision tree: Decision trees are an approach used in supervised machine learning, a technique which uses labelled input and output datasets to train models.
              
              The approach is used mainly to solve classification problems, which is the use of a model to categorise or classify an object.

        Regression

            - Linear regression: Linear Regression is a supervised machine learning algorithm where the predicted output is continuous and has a constant slope.
              
              It's used to predict values within a continuous range, (e.g. sales, price) rather than trying to classify them into categories (e.g. cat, dog).

        Classification & Regression

            - KNN: The abbreviation KNN stands for “K-Nearest Neighbour”. It is a supervised machine learning algorithm. The algorithm can be used to solve both classification and regression problem statements.
              
              The number of nearest neighbours to a new unknown variable that has to be predicted or classified is denoted by the symbol 'K'.

            - Random Forest: Random Forest is a robust machine learning algorithm that can be used for a variety of tasks including regression and classification.
              
              It is an ensemble method, meaning that a random forest model is made up of a large number of small decision trees, called estimators, which each produce their own predictions.
              
              The random forest model combines the predictions of the estimators to produce a more accurate prediction. """
    
    return print(text)


def best_param_clf(X_train, y_train, model):
    """This function allows you to select the best parameters for your machine learning model.
    
    You may choose the model among the following: 'Logistic' = LogisticRegression, 'Kneighbors' = KNeighborsClassifier, 'RandomForest' = RandomForestClassifier, 'SupportVC' = SVC, 'DecisionTree' = DecisionTreeClassifier

    The following libraries should be implemented
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
"""

    if model == "Logistic":
        clf = LogisticRegression()
        parameter_grid = {'C': [0.01, 0.1, 1, 2, 10, 100], 'penalty': ['l1', 'l2'], 'solver' : ['liblinear','lbfgs']}
        gridsearch = GridSearchCV(clf, parameter_grid)
        gridsearch.fit(X_train, y_train)
        return print("The best parameters for the Logistic Regression model are: ", gridsearch.best_params_)
    elif model == "KNeighbors":
        clf = KNeighborsClassifier()
        parameter_grid = {"n_neighbors": [3,5,7,9,11],"weights": ["uniform", "distance"]}
        gridsearch = GridSearchCV(clf, parameter_grid)
        gridsearch.fit(X_train, y_train)
        return print("The best parameters for the K Neighbors Classifier model are: ", gridsearch.best_params_)
    elif model == "RandomForest":
        clf =  RandomForestClassifier()
        parameter_grid = {"n_estimators": [100,120,150,200,300,500],"max_depth": [3,4,5,6,10,15,17],"max_features": ["sqrt", 3, 4]}
        gridsearch = GridSearchCV(clf, parameter_grid)
        gridsearch.fit(X_train, y_train)
        return print("The best parameters for the Random Forest Classifier model are: ", gridsearch.best_params_)
    elif model == "SuportVC":  
        clf = SVC()
        parameter_grid = {"C": [0.01, 0.1, 0.3, 0.5, 1.0, 3, 5.0, 15, 30], "kernel": ["linear", "poly", "rbf"], "degree": [2, 3, 4, 5],"gamma": [0.001, 0.1, "auto", 1.0, 10.0, 30.0]}
        gridsearch = GridSearchCV(clf, parameter_grid)
        gridsearch.fit(X_train, y_train)
        return print("The best parameters for the SVC model are: ", gridsearch.best_params_)
    elif model == "DecisionTree":
        clf = DecisionTreeClassifier()
        parameter_grid = {"max_depth": list(range(1,10))}
        gridsearch = GridSearchCV(clf, parameter_grid)
        gridsearch.fit(X_train, y_train)
        return print("The best parameters for the Decision Tree Classifier model are: ", gridsearch.best_params_)


def classif_metrics(pred, y_test):
    """ 
    This function allows you to have the metrics of your classification model
    
    Syntax: pred(Prediction X_test(y_pred)), y_test
    
    "pred" has to be defined "as is" outside the function
    
    Required Libraries:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
    """
    metrics = ["Accuracy", (accuracy_score(pred, y_test))], ["Precision", (precision_score(pred, y_test))],["Recall", (recall_score(pred, y_test))],["F1", (f1_score(pred, y_test))], ["AUC", (roc_auc_score(pred, y_test))] 
    metrics_df = pd.DataFrame(metrics, columns = ['METRICS', 'Results'])
    metrics_df.set_index("METRICS", inplace= True)
    return print(metrics_df)


def hashing(X_train, name_column, new_clm_1, new_clm_2, new_clm_3, new_clm_4, new_clm_5):
    ''' Con está función se puede tratar una característica categórica de alta cardinalidad para obtener varias columnas númericas sin perder información
    
    Required Libraries:
    from sklearn.feature_extraction import FeatureHasher
    '''

    h = FeatureHasher(n_features=5, input_type='string')
    f = h.transform(X_train[name_column])
    f.toarray()

    hash_df= pd.DataFrame(f.toarray())

    hash_df = hash_df.rename(columns={0:new_clm_1,
                        1:new_clm_2,
                        2:new_clm_3,
                        3:new_clm_4,
                        4:new_clm_5})

    hash_df.set_index(X_train.index,inplace = True)

    X_train = pd.concat([hash_df, X_train],axis = 1 )
    X_train= X_train.drop([name_column], axis=1)
    return X_train
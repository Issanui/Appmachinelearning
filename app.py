from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scikitplot.metrics import plot_roc_curve, plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay,PrecisionRecallDisplay
st.set_option('deprecation.showPyplotGlobalUse', False)
from io import StringIO

def main():
    st.title("Application machine learning pour la prediction du churn")
    st.subheader("GROUPE ISE2 B: DJIBRILLA Issa, BOUWO GLADYS, DEBORA")

    # fonction d'importation des données
    dataset=st.sidebar.file_uploader("Charger les données", type=["csv"])
    if dataset is not None:
        #dataset=dataset.getvalue()
        dataset = StringIO(dataset.getvalue().decode("utf-8"))
    # Affichage des données
        df= pd.read_csv(dataset) 
        df = pd.get_dummies(df)
        df_sample=df.sample(100)
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Jeu des données: Echantillon 100 abservations")
        st.write(df_sample)

    seed = 123

    #Train/test split
    def split(df):
        y = df['CHURN']
        X = df.drop('CHURN', axis=1)
        X = df.drop('ID', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, stratify=y, random_state=seed)
        return X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = split(df)

    Classifier = st.sidebar.selectbox("MODELE", ("Random Forest", "K-NN", "Regression logistique"))

    # Analyse de la performance des modèles
    def plot_perf(graphes):
        if "Confusion matrix" in graphes:
            st.subheader("Matrice de confusion")
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
            st.pyplot()
        if "ROC curve" in graphes:
            st.subheader("Courbe de ROC")
            RocCurveDisplay.from_estimator(model, X_test, y_test)
            st.pyplot()
        if "Precision-Recall curve" in graphes:
            st.subheader("Courbe Precision-Recall")
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
            st.pyplot()
    
    #K-NN
    if Classifier == "K-NN":
        st.sidebar.subheader("Hyperparamètres du modèle")
        n_neighbors=st.sidebar.number_input("Nombre de voisins", 1, 100)
        graphes_perf = st.sidebar.multiselect("Choisir un graphique de performance du modele ML",("Confusion matrix", "ROC curve", "Precision-Recall curve"))
        if st.sidebar.button("Execution", key=Classifier):
            st.subheader("Resultats du K-NN")
            # Initiation d'un object K-NN
            model= KNeighborsClassifier(n_neighbors)
            #Entrainement de l'algorithme
            model.fit(X_train, y_train)
            #Predictions
            y_pred = model.predict(X_test)
            y_probas=model.predict_proba(X_test)
            y_probas=np.concatenate((1-y_probas,y_probas),axis=1)
            # Calcul des metriques de performance
            accuracy=accuracy_score(y_test, y_test)
            precision=precision_score(y_test, y_pred)
            recall=recall_score(y_test, y_pred)
            # Afficher les metriques dans l'application
            st.write("Accuracy:", accuracy.round(2))
            st.write("precision:", precision.round(2))
            st.write("recall:", recall.round(2))
            #Afficher les graphiques de performances
            plot_perf(graphes_perf)
    
    #Regression logistique
    if Classifier == "Regression logistique":
        st.sidebar.subheader("Hyperparamètres du modèle")
        hyp_c=st.sidebar.number_input("Choisir la valeur du parametre de regularisation", 0.01, 10.0)
        n_max_iter = st.sidebar.number_input("ProfondeLe nombre  maximale d'iteration", 100, 1000, step=10)
        graphes_perf = st.sidebar.multiselect("Choisir un graphique de performance du modele ML",("Confusion matrix", "ROC curve", "Precision-Recall curve"))
        if st.sidebar.button("Execution", key=Classifier):
            st.subheader("Resultats de la regression logistique")
            # Initiation d'un object LogisticRegression
            model= LogisticRegression(C=hyp_c,max_iter=n_max_iter, random_state=seed)
            #Entrainement de l'algorithme
            model.fit(X_train, y_train)
            #Predictions
            y_pred = model.predict(X_test)
            y_probas=model.predict_proba(X_test)
            y_probas=np.concatenate((1-y_probas,y_probas),axis=1)
            # Calcul des metriques de performance
            accuracy=accuracy_score(y_test, y_test)
            precision=precision_score(y_test, y_pred)
            recall=recall_score(y_test, y_pred)
            # Afficher les metriques dans l'application
            st.write("Accuracy:", accuracy.round(2))
            st.write("precision:", precision.round(2))
            st.write("recall:", recall.round(2))
            #Afficher les graphiques de performances
            plot_perf(graphes_perf)
        
    #Random Forest
    if Classifier == "Random Forest":
        st.sidebar.subheader("Hyperparamètres du modèle")
        n_estimators=st.sidebar.number_input("Choisir le nombre d'arbres dans la foret", 100, 1000, step=10)
        max_depth = st.sidebar.number_input("Profondeur maximale d'un arbre", 1, 20, step=1)
        bootstrap=st.sidebar.radio("Echantillon bootstrap lors de la creation d'arbres ?", ("True", "False"))
        graphes_perf = st.sidebar.multiselect("Choisir un graphique de performance du modele ML",("Confusion matrix", "ROC curve", "Precision-Recall curve"))
        if st.sidebar.button("Execution", key=Classifier):
            st.subheader("Resultats de Random Forest")
            # Initiation d'un object RandomForestClassifier
            model= RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth, bootstrap=bool(bootstrap), random_state=seed)
            #Entrainement de l'algorithme
            model.fit(X_train, y_train)
            #Predictions
            y_pred = model.predict(X_test)
            y_probas=model.predict_proba(X_test)
            y_probas=np.concatenate((1-y_probas,y_probas),axis=1)
            # Calcul des metriques de performance
            accuracy=accuracy_score(y_test, y_test)
            precision=precision_score(y_test, y_pred)
            recall=recall_score(y_test, y_pred)
            # Afficher les metriques dans l'application
            st.write("Accuracy:", accuracy.round(2))
            st.write("precision:", precision.round(2))
            st.write("recall:", recall.round(2))
            #Afficher les graphiques de performances
            plot_perf(graphes_perf)
if __name__ == '__main__':
    main()

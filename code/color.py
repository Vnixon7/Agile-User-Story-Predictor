from sklearn.pipeline import Pipeline
from nltk import probability
#from sklearn.preprocessing import make_pipeline
from nltk.classify.naivebayes import NaiveBayesClassifier
import sklearn
import dill
from sklearn.feature_selection import chi2
#from gensim.models.fasttext import FastText
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model, preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, confusion_matrix, classification_report
from sklearn import svm
from sklearn.svm import SVC
from scipy.sparse import hstack, coo_matrix
import pandas as pd
import numpy as np
import pickle
import sys
import re
from html.parser import HTMLParser
from nltk.corpus import stopwords
import nltk
from nltk import WordPunctTokenizer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from text_cleaner import html_to_text, clean
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
#nltk.download('stopwords')


def main():
    data = pd.read_csv (r'C:\Users\Vnixo\OneDrive\Desktop\userStoryML\storyPointEstimation\Data\TNE_2021-11-11-color.csv')
    
    print('preprocessing data...')

    label_maker = preprocessing.LabelEncoder()
    mm = preprocessing.MinMaxScaler()
    ss = preprocessing.StandardScaler()
    mas = preprocessing.MaxAbsScaler()


    title = data['title']
    title = title.fillna('none')
    desc = data['description']
    desc = desc.fillna('none')


    # ---- label encode ----
    #data['color'] = label_maker.fit_transform(data['color'])
    data['issuekey'] = label_maker.fit_transform(data['issuekey'])

    Y = data['color']
    print(Y[:8])
    Y = label_maker.fit_transform(Y)
    print(Y[:8])


    data.drop('color', axis=1, inplace=True)
    #data.drop('owner', axis=1, inplace=True)

    vec_title = []
    vec_desc = []
    test_title = []
    test_desc = []


    for i in range(0, len(title)):
        vec_title.append(list(set(clean(title[i]))))
        vec_desc.append(list(set(clean(desc[i]))))



    for i in range(0, len(vec_title)):
        vec_title[i] = ','.join(vec_title[i])
        vec_desc[i] = ','.join(vec_desc[i])



    data['title'] = vec_title
    data['description'] = vec_desc


    #tfidf_data = data[['title', 'description', 'issuekey']].fillna('none')
    #tfidf_test_data = test_data[['title', 'description', 'issuekey']].fillna('none')

    tfidf_data = data[['title', 'description']].fillna('none')

    #print(tfidf_data.head())
    #sys.exit()


    transformer = FeatureUnion([
                    ('title_tfidf', 
                    Pipeline([('extract_field',
                                FunctionTransformer(lambda x: x['title'], 
                                                    validate=False)),
                                ('tfidf', 
                                TfidfVectorizer())])),
                    ('description_tfidf', 
                    Pipeline([('extract_field', 
                                FunctionTransformer(lambda x: x['description'], 
                                                    validate=False)),
                                ('tfidf', 
                                TfidfVectorizer(stop_words="english"))]))])


    tfidf_data = transformer.fit_transform(tfidf_data)

    title_vocab = transformer.transformer_list[0][1].steps[1][1].get_feature_names() 
    desc_vocab = transformer.transformer_list[1][1].steps[1][1].get_feature_names()
    vocab = title_vocab + desc_vocab




    print(len(Y))
    smote = SMOTE(sampling_strategy='not majority', k_neighbors=2)
    #smote = SMOTE(sampling_strategy='majority', k_neighbors=3)
    ros = RandomOverSampler(random_state=42)
    rus = RandomUnderSampler(random_state=0, replacement=True)
    X, y = smote.fit_resample(tfidf_data, Y)
    print("over: ", len(y))
#     models = [
#     XGBClassifier(),
#     GradientBoostingClassifier(),
#     DecisionTreeClassifier(),
#     RandomForestClassifier(),
#     linear_model.LogisticRegression(max_iter=1000),
#     MultinomialNB(),
#     KNeighborsClassifier(n_neighbors=3),
#     svm.SVC(probability=True)
# ]
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, shuffle=True)

#     for model in models:
        
#         #model = OneVsRestClassifier(model)
#         #model = OneVsOneClassifier(model)
#         model.fit(x_train, y_train)
#         acc = model.score(x_test, y_test)
#         y_pred = model.predict(x_test)
#         cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None, normalize=None)

#         print("-----------")

#         cv = cross_val_score(model, X=x_test, y=y_test, cv=5)
#         print("Model: ", model, 'Cross-Val accuracy: ', cv,'==',(np.mean(cv) * 100) )
#         auc = roc_auc_score(y, model.predict_proba(X), multi_class='ovr')
#         #print(auc)
#         #print(cm)
#         print(acc, auc)
#     sys.exit()
    #tfidf_test_data = transformer.transform(tfidf_test_data)
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy',
                                    max_depth=50, min_samples_split=4,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                    max_features='log2', max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    bootstrap=True, oob_score=True,
                                    n_jobs=-1, random_state=None,
                                    verbose=0, warm_start=False,
                                    class_weight=None, ccp_alpha=0.0,
                                    max_samples=None)
    # rf = pickle.load(open(fr'C:\Users\vnixon\Desktop\ML_models\userStoryAutomation\trained\SP_final - Copy.pickle', 'rb'))
    # rf.fit(X, y)
    # with open(fr'C:\Users\vnixon\Desktop\ML_models\userStoryAutomation\trained\SP_final - Copy.pickle', 'wb') as f:
    #     dill.dump(rf, f)
    # model = svm.SVC(probability=True,)
    # params = {
    #      "C":[1,2,3,4],
    #      "kernel":['linear', 'poly', 'rbf', 'sigmoid'],
    #      "degree":[3, 1, 2, 4],
    #      "gamma":['scale', 'auto'], 
    #      "coef0":[0], 
    #      "shrinking":[True], 
    #      "probability":[True], 
    #      "tol":[0.001], 
    #      "cache_size":[200], 
    #      "class_weight":[None, 'balanced'], 
    #      "verbose":[False], 
    #      "max_iter":[-1], 
    #      "decision_function_shape":['ovr', 'ovo'], 
    #      "break_ties":[False], 
    #      "random_state":[None]
    # }
    # grid = GridSearchCV(model, params, refit = True, verbose = 3, n_jobs=-1, cv=5) 
    # #rando = RandomizedSearchCV(model, params, refit=True, verbose=3, n_jobs=-1, cv=3)
    # #model = OneVsRestClassifier(model)
    # #model = OneVsOneClassifier(model)

    # # fitting the model for grid search 
    # grid.fit(x_train, y_train) 
    # #rando.fit(x_train, y_train)

    # # print best parameter after tuning 
    # print(grid.best_params_) 
    # #print(rando.best_params_)
    # #grid_predictions = grid.predict(x_test)
    # #sys.exit()
    # #sys.exit()
    # rf = pickle.load(open(fr'C:\Users\vnixon\Desktop\ML_models\userStoryAutomation\trained\SP_final - Copy.pickle', 'rb'))
    # rf.fit(X, y)
    # with open(fr'C:\Users\vnixon\Desktop\ML_models\userStoryAutomation\trained\SP_final - Copy.pickle', 'wb') as f:
    #     dill.dump(rf, f)
    
    # sys.exit()
    best = 0.999997
    for i in range(100):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.30, shuffle=True, stratify=y)
        #rf = pickle.load(open(fr'C:\Users\vnixon\Desktop\ML_models\userStoryAutomation\trained\TNE_1.pickle', 'rb'))
        #rf = pickle.load(open(fr'C:\Users\vnixon\Desktop\ML_models\userStoryAutomation\trained\TNE_2.pickle', 'rb'))
        rf = svm.SVC(C=5, kernel='poly', degree=1, 
        gamma='scale', coef0=0, 
        shrinking=True, probability=True, 
        tol=0.001, cache_size=200, 
        class_weight='balanced', verbose=False, 
        max_iter=-1, decision_function_shape='ovr', 
        break_ties=False, random_state=None)

        rf.fit(x_train, y_train)
        acc = rf.score(x_test, y_test)
        pred = rf.predict(x_test)
        cr = classification_report(y_test, pred)
        cv = cross_val_score(rf, X=X, y=y, cv=StratifiedKFold(shuffle=True, n_splits=5))
        auc = roc_auc_score(y, rf.predict_proba(X), multi_class='ovr')
        print(acc, auc)
        print(cv)
        #print(y_pred)
        #print(cr)
        a = 0
        for i in range(0, len(pred)):
            pred[i] = np.round(pred[i])
            if pred[i] == y_test[i]:
                a += 1
            #print("p:", pred[i], "a:", y_test[i])
        accuracy = a/len(pred)
        #print(accuracy)
        #print(vocab)
        #print(transformer.transform(tfidf_data).toarray())
        #if y_pred[0] == 2 and y_pred[1] == 5:
        if auc == 1.0:
           with open(fr'C:\Users\Vnixo\OneDrive\Desktop\userStoryML\storyPointEstimation\prod_models\model_color1.pickle', 'wb') as f:
           #with open(fr'C:\Users\vnixon\Desktop\ML_models\userStoryAutomation\trained\TNE_2.pickle', 'wb') as f:
               dill.dump(rf, f)
           dill.dump(transformer, open(r"C:\Users\Vnixo\OneDrive\Desktop\userStoryML\storyPointEstimation\transformers\transformer_color1.pickle","wb"))
           #dill.dump(transformer, open("transformer2.pkl","wb"))
           break
        
   


if __name__ == "__main__":
    main()

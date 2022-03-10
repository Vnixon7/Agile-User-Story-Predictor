import sklearn
from sklearn import linear_model, preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, confusion_matrix
from scipy.sparse import hstack, coo_matrix
import pandas as pd
import numpy as np
import pickle
import sys
from html.parser import HTMLParser
from text_cleaner import html_to_text, clean
#nltk.download('stopwords')
import dill

def get_app(output):
    new_output = []
    apps = {0:"|BigFix",
            1:"|Bright Idea|Cognos|ICC/ITEX|IT Gateway|Miscellaneous|Policies and Standards|PPM|Rally|SharePoint Apps|Supplier Management|Enterprise One", 
            2:"|Cognos|PPM", 
            3:"|Enterprise One", 
            4:"|Enterprise One|SmartTrack|BigFix", 
            5:"|ICC/ITEX",
            6:"|PPM",
            7:"|Rally",
            8:"|Rally|Enterprise One", 
            9:"|Rally|Enterprise One|SmartTrack|BigFix",
            10:"|SmartTrack"
            }
            
    for i in output:
        new_output.append(apps[i])
    return new_output

def get_owner(output):
    new_output = []
    owners = {0:"Person_1",
              1:"Person_2",
              2:"Person_3",
              3:"Person_4",
              4:"Person_5",
              5:"Person_6"}

    for i in output:
        new_output.append(owners[i])
    return new_output

def get_color(output):
    new_output = []
    colors = {
              0:'#21a2e0',
              1:'#4a1d7e',
              2:'#848689',
              3:'#df1a7b',
              4:'#ee6c19',
              5:'#f9a814'
                        }
    for i in output:
        new_output.append(colors[i])
    return new_output


if __name__ == "__main__":
    
    test_data = pd.read_csv (r'C:\Users\Vnixo\OneDrive\Desktop\userStoryML\storyPointEstimation\Data\predictions1.csv')
    sp = pickle.load(open(fr'C:\Users\Vnixo\OneDrive\Desktop\userStoryML\storyPointEstimation\prod_models\model_sp1.pickle', 'rb'))
    color = pickle.load(open(fr'C:\Users\Vnixo\OneDrive\Desktop\userStoryML\storyPointEstimation\prod_models\model_color1.pickle', 'rb'))
    owner = pickle.load(open(fr'C:\Users\Vnixo\OneDrive\Desktop\userStoryML\storyPointEstimation\prod_models\model_owner1.pickle', 'rb'))
    app = pickle.load(open(fr'C:\Users\Vnixo\OneDrive\Desktop\userStoryML\storyPointEstimation\prod_models\model_app1.pickle', 'rb'))

    transformer1 = dill.load(open(r"C:\Users\Vnixo\OneDrive\Desktop\userStoryML\storyPointEstimation\transformers\transformer_sp1.pickle","rb"))
    transformer2 = dill.load(open(r"C:\Users\Vnixo\OneDrive\Desktop\userStoryML\storyPointEstimation\transformers\transformer_color1.pickle","rb"))
    transformer3 = dill.load(open(r"C:\Users\Vnixo\OneDrive\Desktop\userStoryML\storyPointEstimation\transformers\transformer_owner1.pickle","rb"))
    transformer4 = dill.load(open(r"C:\Users\Vnixo\OneDrive\Desktop\userStoryML\storyPointEstimation\transformers\transformer_app1.pickle","rb"))


    label_maker = preprocessing.LabelEncoder()
    mm = preprocessing.MinMaxScaler()
    ss = preprocessing.StandardScaler()
    mas = preprocessing.MaxAbsScaler()


    t_title = test_data['title']
    t_title = t_title.fillna('none')
    t_desc = test_data['description']
    t_desc = t_desc.fillna('none')


    vec_title = []
    vec_desc = []
    test_title = []
    test_desc = []


    for i in range(0, len(t_title)):
        test_title.append(list(set(clean(t_title[i]))))
        test_desc.append(list(set(clean(t_desc[i]))))

    for i in range(0, len(vec_title)):
        vec_title[i] = ','.join(vec_title[i])
        vec_desc[i] = ','.join(vec_desc[i])


    for i in range(0, len(test_title)):
        test_title[i] = ','.join(test_title[i])
        test_desc[i] = ','.join(test_desc[i])

    test_data['title'] = test_title
    test_data['description'] = test_desc

    tfidf_test_data = test_data[['title', 'description']].fillna('none')

    tfidf_test_data1 = transformer1.transform(tfidf_test_data)
    tfidf_test_data2 = transformer2.transform(tfidf_test_data)
    tfidf_test_data3 = transformer3.transform(tfidf_test_data)
    tfidf_test_data4 = transformer4.transform(tfidf_test_data)
    #tfidf_test_data5 = transformer5.transform(tfidf_test_data)


    sp = sp.predict(tfidf_test_data1)
    #sp2 = sp2.predict(tfidf_test_data1)
    color = get_color(color.predict(tfidf_test_data2))
    #color2 = get_color(color2.predict(tfidf_test_data5))
    owner = get_owner(owner.predict(tfidf_test_data3))
    app = get_app(app.predict(tfidf_test_data4))


    for i in range(0, len(sp)):
        print(f"MODEL1 -> App: {app[i]}  Color: {color[i]}  Owner: {owner[i]}  StoryPoint: {sp[i]}")
    print("-----------")
    #for i in range(0, len(sp)):
    #    print(f"MODEL2 -> App: {app[i]}  Color: {color2[i]}  Owner: {owner[i]} StoryPoint: {sp2[i]}")



    #print(tfidf_data.head())
    #sys.exit()

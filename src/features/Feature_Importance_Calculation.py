import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import multiprocessing
import time

starttime = time.time()

def read_file(path):
    
    file_dict = pd.read_pickle(path)
    df = pd.DataFrame(file_dict.items(),columns=['Name','Feature_Value'])
    columns = []
    for i in range(1024):
        columns = columns + [str(i)]
    df_final = pd.DataFrame(df['Feature_Value'].to_list(),columns = columns)
    df_final['Name'] = df['Name']
  
    return df_final

def dataset_labels(csv_file_path):

    labels_df = pd.read_csv(csv_file_path)
    labels_df = labels_df.dropna()
    labels_df.astype(int)
    labels_dict = {}
    files_list = labels_df['Sample ID'].to_list()
    grade = labels_df['Sample Grade'].to_list()

    for i in range(len(files_list)):
        labels_dict[int(files_list[i])] = int(grade[i])
    
    return labels_dict

df = read_file('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/kimianet_features/FineTuned_Model_Features_dict.pickle')

labels_dict = dataset_labels('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Data.csv')

df['Grade'] = ""
i = 0
list_Temp = df['Name'].to_list()
label_list = []

for patch in list_Temp:
    label = labels_dict[int(((patch).split(' ')[1]).split('.')[0])]
    # print(label)
    label_list = label_list + [label]
    # print(len(label_list))

df['Grade'] = label_list
df.to_csv('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/kimianet_features/features.csv')
df_final = df
result_df = pd.DataFrame()
accuracy_list = []
feature_number = []
# result_list = []

def feature_analyzer(i,return_dict):

    print('Analyzing Feature : {}'.format(i))
    X = df_final.iloc[:,[i]].values

    y = df_final['Grade'].values
    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.30, random_state=0)
   
    sc_X = StandardScaler() 
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)

    classifier = SVC(kernel='linear')

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    score = accuracy_score(y_test,y_pred)

    # result_list[i] = score
    if score > 0.6:
        print(score, i)
        
    return_dict[i] = score

    print('Analysis of Feature {} is complete'.format(i))
    
processes = []
result_list = [None] * 1024
manager = multiprocessing.Manager()
return_dict = manager.dict()

for i in range(1024):

    # feature_analyzer(i)
    p = multiprocessing.Process(target = feature_analyzer,args=(i,return_dict))
    processes.append(p)
    p.start()

for process in processes:
    process.join()

print('That took {} seconds'.format(time.time() - starttime))

print(len(return_dict))
print(result_list[0])
result_df = pd.DataFrame((return_dict.values()),columns=['Accuracy'])
# result_df = pd.DataFrame(result_df['temp'].to_list(),columns = ['Accuracy','Feature Number'])

# result_df['feature_number'] = feature_number
# result_df['accuracy'] = accuracy_list
print('Saving The Dataframe')

result_df.to_csv('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/feature_accuracy.csv')
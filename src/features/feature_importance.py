import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import multiprocessing
import time

parser = argparse.ArgumentParser(description='Script for Feature Importance Calculation Using Support Vector Machines')
parser.add_argument('dst',help= 'Directory to store the final Results CSV File')
parser.add_argument('features',help = 'Path to the Extracted Feature File')
parser.add_argument('labels',help = 'Path to the Data Labels CSV File')
args = parser.parse_args()
config = vars(args)


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



def feature_analyzer(i,df_final,return_dict):

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

    if score > 0.6:
        print(score, i)
        
    return_dict[i] = score
    print('Analysis of Feature {} is complete'.format(i))
    


if __name__ == '__main__':


    dst_file = config['dst']
    feature_file = config['feature']
    labels_csv = config['labels']

    starttime = time.time()
   
    df = read_file(feature_file)
    labels_dict = dataset_labels(labels_csv)

    df['Grade'] = ""
    # i = 0
    list_Temp = df['Name'].to_list()
    label_list = []

    for patch in list_Temp:
        label = labels_dict[int(((patch).split(' ')[1]).split('.')[0])]
        label_list = label_list + [label]

    df['Grade'] = label_list
  

    result_df = pd.DataFrame()
    processes = []
    result_list = [None] * 1024
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for i in range(1024):

        p = multiprocessing.Process(target = feature_analyzer,args=(i,df,return_dict))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print('That took {} seconds'.format(time.time() - starttime))
    result_df = pd.DataFrame((return_dict.values()),columns=['Accuracy'])
    print('Saving The Dataframe')

    final_path = os.path.join(dst_file,'feature_accuracy.csv')
    result_df.to_csv(final_path)


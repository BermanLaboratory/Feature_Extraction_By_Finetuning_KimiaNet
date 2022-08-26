import os
import argparse

parser = argparse.ArgumentParser(description='Script to Remove Empty Tiles from the Images Using File Size')
parser.add_argument("src",help = 'Source to run the Script on - Can be a single folder or directory with muliple folders')
parser.add_argument('--file_size',type = int,default = 1,help='The Tiles having file size less than this argument will be removed. (Pass an Integer in mb)')
args = parser.parse_args()
config = vars(args)

if __name__ == '__main__':


    size = config['file_size']
    folder_paths = []
    with os.scandir(config['src']) as folder_list:
        for folder in folder_list:
            if(folder.is_dir()):
                folder_paths.append(folder.path)
    
    if (len(folder_paths) == 0):

        print('Source Is A Single Folder')

        count = 0
        
        with os.scandir(config['src']) as files:
            for file in files:
                if (os.path.getsize(file.path)/(1024*1024)) <= size :
                    os.remove(file.path)
                    count = count + 1
        folder_name = config['src'].split('/')[-1]
        print('{} Files Removed from folder {}'.format(count,folder_name))

    else:

        print('There are {} folders in the directory to process'.format(len(folder_paths)))
        for folder in folder_paths:
            count = 0
            folder_name = folder.split('/')[-1]
            print('File Removal Started for folder : {}'.format(folder_name))
            with os.scandir(folder) as files:
                for file in files:
                    if (os.path.getsize(file.path)/(1024*1024)) <= size :
                        os.remove(file.path)
                        count = count + 1
                        if count % 100 == 0 :
                            print('         {} files removed'.format(count))
         
            print('{} Files Removed from folder {}'.format(count,folder_name))


    print('ALL FILES HAVING FILE SIZE <= {} mb ARE SUCCESSFULLY REMOVED.'.format(size))
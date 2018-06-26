from pull_data_mod import Pull
import os
import csv
import shutil
max_files=[None,None]
compact=1
bd_compact=0
print '------Below are labels------'
#print d.labels
def data_gen(d,filename,labelname,label):
    with open (filename,'w+') as datacsv:
        csvwriter = csv.writer(datacsv)
        for  data in d.data:
            csvwriter.writerow(data)
        datacsv.close 

    with open (labelname,'w+') as labelcsv:
        csvwriter = csv.writer(labelcsv)
        for  i in range(len(d.data)):
            csvwriter.writerow(label)
        labelcsv.close
pos_dir='good/'
neg_dir='bad/'
filelist=os.listdir('.')
for item in filelist:
    if os.path.splitext(item)[1]=='.gz': 
        print item
        isExists=os.path.exists(item[:-3])
        if not isExists:
            os.makedirs(item[:-3])   # make new dir
            shutil.copyfile(item,item[:-3]+'/'+item)  # copy to the new dir 
        else:
            print 'The dir exists'
            
for item in filelist:
    if os.path.isdir(item):
        neg_dir= item+'/'
        fn=item
        label = "_9"
        # tmp PL IPT BD TLS
        print 'staus 1 start deal with',item
        types=[0,1,2,3,4]
        filename = "data"+fn+label+"_1.csv"
        labelname = "label"+fn+label+"_1.csv"
        if not os.path.exists(filename):
            d=Pull(pos_dir, neg_dir, types, compact, max_files, bd_compact)
            data_gen(d,filename,labelname,label)
            print 'staus 1 done deal with ',item
        else:
            print 'csv file already exists.'

'''
fn="out2018-01-29_win7"
label = "_9"
# tmp PL IPT BD TLS
print 'staus 1 start'
types=[0,1,2,3,4]
filename = "data"+fn+label+"_1.csv"
labelname = "label"+fn+label+"_1.csv"
d=Pull(pos_dir, neg_dir, types, compact, max_files, bd_compact)
data_gen(d,filename,labelname,label)
print 'staus 1 done'
'''

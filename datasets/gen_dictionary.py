# -*- coding: utf-8 -*-
label_txt=["data/train_caption.txt","data/test-caption.txt"]

data=[]
for txt in label_txt:
    with open(txt,"r") as fp:
        data+=fp.readlines()
    
labels=[]
for line in data:
    line=line.strip().split("\t")
    labels+=line[1].split(" ")
    
labels=list(set(labels))

dictionary="data/dictionary1.txt"
fp=open(dictionary,"w",encoding="utf-8")

# sos   0
# eos	1
fp.write("sos 0\n")
fp.write("eos 1\n")

j=1
for i in range(len(labels)):
    fp.write("{} {}\n".format(labels[i],i+2))
    j+=1

fp.write("<eol> {}\n".format(j+1))
fp.close()
import numpy as np
import string

def load_glove_model():
    print("Loading Glove Model")
    glove_model = {}
    with open("glove.6B.300d.txt",'r') as f:
        for line in f:
            try:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
            except:
                print(split_line)
                break
    print(f"{len(glove_model)} words loaded!")
    return glove_model

def camel_case(example):      
    if  any(x in example for x  in string.punctuation)==True:
        return False
    else:
        if any(list(map(str.isupper, example[1:-1])))==True:
            return True
        else:
            return False    

def camel_case_split(word):
    idx = list(map(str.isupper, word))
    case_change = [0]
    for (i, (x, y)) in enumerate(zip(idx, idx[1:])):
        if x and not y:  
            case_change.append(i)
        elif not x and y:  
            case_change.append(i+1)
    case_change.append(len(word))
    return [word[x:y] for x, y in zip(case_change, case_change[1:]) if x < y]

def tokenizer(mylist):
    tokenized_list=[]
    for word in mylist:

        if '_'  in word:
            splitted_word=word.split('_')
            for elem in splitted_word:
                if camel_case(elem):
                    elem=camel_case_split(elem)
                    for el1 in elem:
                        tokenized_list.append(el1.lower())
                else:    
                    tokenized_list.append(elem.lower())
        elif '-' in word:
            hyp_word=word.split('-')
            for i in hyp_word:
                if camel_case(i):
                    i=camel_case_split(i)
                    for el2 in i:
                        tokenized_list.append(el2.lower())
                else: 
                    tokenized_list.append(i.lower())
        elif camel_case(word):
            word=camel_case_split(word)
            for el in word:
                tokenized_list.append(el.lower())
        else:
            tokenized_list.append(word.lower())
    return(tokenized_list)

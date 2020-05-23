# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## 1.生成词向量
# 
# 下载GitHub项目：[https://github.com/stanfordnlp/GloVe/archive/master.zip](https://github.com/stanfordnlp/GloVe/archive/master.zip)
# 
# 解压后，进入目录执行
# 
# make
# 
# 进行编译操作。
# 
# 然后执行 sh demo.sh 进行训练并生成词向量文件：vectors.txt和vectors.bin
# 
# ## 2. 词向量生成模型并加载

# %%
import shutil
import gensim


# %%
def getFileLineNums(filename):  
    f = open(filename,'r')  
    count = 0  
  
    for line in f:  
          
        count += 1  
    return count
 
def prepend_line(infile, outfile, line):  
    """ 
    Function use to prepend lines using bash utilities in Linux. 
    (source: http://stackoverflow.com/a/10850588/610569) 
    """  
    with open(infile, 'r') as old:  
        with open(outfile, 'w') as new:  
            new.write(str(line) + "\n")  
            shutil.copyfileobj(old, new)
            
def prepend_slow(infile, outfile, line):  
    """ 
    Slower way to prepend the line by re-creating the inputfile. 
    """  
    with open(infile, 'r') as fin:  
        with open(outfile, 'w') as fout:  
            fout.write(line + "\n")  
            for line in fin:  
                fout.write(line) 
                
def load(filename):  
      
    # Input: GloVe Model File  
    # More models can be downloaded from http://nlp.stanford.edu/projects/glove/  
    # glove_file="glove.840B.300d.txt"  
    glove_file = filename  
      
    dimensions = 50  
      
    num_lines = getFileLineNums(filename)  
    # num_lines = check_num_lines_in_glove(glove_file)  
    # dims = int(dimensions[:-1])  
    dims = 50  
      
    print(num_lines)  
        #  
        # # Output: Gensim Model text format.  
    gensim_file='glove_model.txt'  
    gensim_first_line = "{} {}".format(num_lines, dims)  
        #  
        # # Prepends the line.  
    #if platform == "linux" or platform == "linux2":  
    prepend_line(glove_file, gensim_file, gensim_first_line)  
    #else:  
    #    prepend_slow(glove_file, gensim_file, gensim_first_line)  
      
        # Demo: Loads the newly created glove_model.txt into gensim API.  
    model=gensim.models.KeyedVectors.load_word2vec_format(gensim_file,binary=False) #GloVe Model  
      
    model_name = gensim_file[6:-4]  
          
    model.save(model_name)  
      
    return model

if __name__ == '__main__':  
    myfile='GloVe-master/vectors.txt'
    model = load(myfile)
    
    #############   模型加载  #######################
    # model_name='glove_model.txt'
    # model = gensim.models.KeyedVectors.load(model_name)  
 
    print(len(model.vocab)) 
 
    word_list = [u'to',u'one']  
   
    for word in word_list:  
        print(word,'--')
        for i in model.most_similar(word, topn=10):  
            print(i[0],i[1])
        print('')


# %%



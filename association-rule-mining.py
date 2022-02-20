#code for Association Rule Mining

import pandas as pd
df = pd.read_csv(r'C:\Users\mimis\Documents\arm-data.csv')
df
df.columns
df.values
data = list(df["products"].apply(lambda x:x.split(',')))
data

#To receive transaction encoder and support packets, we imported mlxtend and apriori
import mlxtend
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data,columns=te.columns_)
df
df.replace({False: 0, True: 1}, inplace=True)
from mlxtend.frequent_patterns import apriori
df1 = apriori(df,min_support=0.01,use_colnames=True)
df1
df1.sort_values(by="support",ascending=False)
df1['length'] = df1['itemsets'].apply(lambda x:len(x))
df1
df1[(df1['length']==2) & (df1['support']>=0.05)]
df.shape
first = pd.DataFrame(df.sum() / df.shape[0], columns = ["Support"]).sort_values("Support", ascending = False)
first
first[first.Support >= 0.0025]

#To receive iterations, we imported itertools
import itertools
second = list(itertools.combinations(first.index, 2))
second = [list(i) for i in second]
second[:10]
value = []
for i in range(0, len(second)):
    temp = df.T.loc[second[i]].sum() 
    temp = len(temp[temp == df.T.loc[second[i]].shape[0]]) / df.shape[0]
    value.append(temp)
secondIteration = pd.DataFrame(value, columns = ["Support"])
secondIteration["index"] = [tuple(i) for i in second]
secondIteration['length'] = secondIteration['index'].apply(lambda x:len(x))
secondIteration = secondIteration.set_index("index").sort_values("Support", ascending = False)
secondIteration = secondIteration[secondIteration.Support > 0.0001]
def ar_iterations(data, num_iter = 1, support_value = 0.1, iterationIndex = None):
    
    # Next Iterations
    def ar_calculation(iterationIndex = iterationIndex): 
        # Calculation of support value
        value = []
        for i in range(0, len(iterationIndex)):
            result = data.T.loc[iterationIndex[i]].sum() 
            result = len(result[result == data.T.loc[iterationIndex[i]].shape[0]]) / data.shape[0]
            value.append(result)
        # Bind results
        result = pd.DataFrame(value, columns = ["Support"])
        result["index"] = [tuple(i) for i in iterationIndex]
        result['length'] = result['index'].apply(lambda x:len(x))
        result = result.set_index("index").sort_values("Support", ascending = False)
        # Elimination by Support Value
        result = result[result.Support > support_value]
        return result    
    
    # First Iteration
    first = pd.DataFrame(df.T.sum(axis = 1) / df.shape[0], columns = ["Support"]).sort_values("Support", ascending = False)
    first = first[first.Support > support_value]
    first["length"] = 1
    
    if num_iter == 1:
        res = first.copy()
        
    # Second Iteration
    elif num_iter == 2:
        
        second = list(itertools.combinations(first.index, 2))
        second = [list(i) for i in second]
        res = ar_calculation(second)
        
    # All Iterations > 2
    else:
        nth = list(itertools.combinations(set(list(itertools.chain(*iterationIndex))), num_iter))
        nth = [list(i) for i in nth]
        res = ar_calculation(nth)
    
    return res

#Defining iterations with association rule
iteration1 = ar_iterations(df, num_iter=1, support_value=0.1)
iteration1
iteration2 = ar_iterations(df, num_iter=2, support_value=0.01)
iteration2
iteration3 = ar_iterations(df, num_iter=3, support_value=0.00025,
              iterationIndex=iteration2.index)
iteration3
iteration4 = ar_iterations(df, num_iter=4, support_value=0.00023,
              iterationIndex=iteration3.index)
iteration4
freq_items = apriori(df, min_support = 0.1, use_colnames = True, verbose = 1)
freq_items.sort_values("support", ascending = False)
from mlxtend.frequent_patterns import association_rules

rules_ap = association_rules(freq_items, metric="confidence", min_threshold=0.8)
rules_fp = association_rules(freq_items, metric="confidence", min_threshold=0.8)

df_ar = association_rules(freq_items, metric = "confidence", min_threshold = 0.5)
df_ar

#Association Table Results
df_ar = association_rules(freq_items, metric = "confidence", min_threshold = 0.5)
df_ar
df_ar[(df_ar.support > 0.15) & (df_ar.confidence > 0.5)].sort_values("confidence", ascending = False)


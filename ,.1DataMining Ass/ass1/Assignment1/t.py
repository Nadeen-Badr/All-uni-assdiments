import pandas as pd

# Load the sample transaction data from b.csv
transaction_df = pd.read_csv('b.csv')

def prune(data, supp):
    df = data[data.supp_count >= supp]
    return df

def count_itemset(transaction_df, itemsets):
    count_item = {}
    for item_set in itemsets:
        for idx, row in transaction_df.iterrows():
            items = row['Items'].split(',')
            set_A = set(item_set)
            set_B = set(items)
            if set_B.intersection(set_A) == set_A:
                if item_set in count_item.keys():
                    count_item[item_set] += 1
                else:
                    count_item[item_set] = 1

    data = pd.DataFrame()
    data['item_sets'] = count_item.keys()
    data['supp_count'] = count_item.values()

    return data

def count_item(trans_items):
    count_ind_item = {}
    for idx, row in trans_items.iterrows():
        items = row['Items'].split(',')
        for item in items:
            if item in count_ind_item.keys():
                count_ind_item[item] += 1
            else:
                count_ind_item[item] = 1

    data = pd.DataFrame()
    data['item_sets'] = count_ind_item.keys()
    data['supp_count'] = count_ind_item.values()
    data = data.sort_values('item_sets')
    return data

def join(list_of_items):
    itemsets = []
    i = 1
    for entry in list_of_items:
        proceding_items = list_of_items[i:]
        for item in proceding_items:
            if(type(item) is str):
                if entry != item:
                    tuples = (entry, item)
                    itemsets.append(tuples)
            else:
                if entry[0:-1] == item[0:-1]:
                    tuples = entry+item[1:]
                    itemsets.append(tuples)
        i = i+1
    if(len(itemsets) == 0):
        return None
    return itemsets

def apriori(trans_data, supp=3, con=0.5):
    freq = pd.DataFrame()
    
    df = count_item(trans_data)
   
    while(len(df) != 0):
        
        df = prune(df, supp)
    
        if len(df) > 1 or (len(df) == 1 and int(df.supp_count >= supp)):
            freq = df
        
        itemsets = join(df.item_sets)
    
        if(itemsets is None):
            return freq
    
        df = count_itemset(trans_data, itemsets)
    return df

freq_item_sets = apriori(transaction_df, 2)  # Lower support threshold
print(freq_item_sets)

def calculate_conf(value1, value2):
    return round(int(value1)/int(value2) * 100, 2)

def strong_rules(freq_item_sets, threshold):

    confidences = {}
    for row in freq_item_sets.item_sets:
        for i in range(len(row)):
            for j in range(len(row)):
                 if i != j:
                    tuples = (row[i], row[j])
                    conf = calculate_conf(freq_item_sets[freq_item_sets.item_sets == row].supp_count, count_item(transaction_df)[count_item(transaction_df).item_sets == row[i]].supp_count)
                    confidences[tuples] = conf

        
    conf_df = pd.DataFrame()
    conf_df['item_set'] = confidences.keys()
    conf_df['confidence'] = confidences.values()

    return conf_df[conf_df.confidence >= threshold]

strong_rules(freq_item_sets, 10.0)  # Lower confidence threshold

from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator

def interesting_rules(freq_item_sets):
    
    lifts = {}
    prob_of_items = []
  
    for row in freq_item_sets.item_sets:
        num_of_items = len(row)
        
        prob_all = freq_item_sets[freq_item_sets.item_sets == row].supp_count / len(transaction_df)
        for i in range(num_of_items):
            prob_of_items.append(count_item(transaction_df)[count_item(transaction_df).item_sets == row[i]].supp_count / len(transaction_df))
        
        lifts[row] = round(float(prob_all / reduce(operator.mul, (np.array(prob_of_items)), 1)), 2)
        
        prob_of_items = []
        
    
    lifts_df = pd.DataFrame()
    lifts_df['Rules'] = lifts.keys()
    lifts_df['lift'] = lifts.values()
    
    return lifts_df

int_rules = interesting_rules(freq_item_sets)
print(int_rules)
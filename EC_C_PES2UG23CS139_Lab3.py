import torch

def get_entropy_of_dataset(data: torch.Tensor):
    """Calculate the entropy of the entire dataset"""
    labels = [row[-1].item() for row in data]     
    def entropy(probs):
        return -torch.sum(probs * torch.log2(probs))    
    unique_labels = list(set(labels))                  
    probs = [labels.count(lbl) / len(labels) for lbl in unique_labels]
    return entropy(torch.tensor(probs)).item()


def get_avg_info_of_attribute(data: torch.Tensor, attr: int):
    """Return avg_info of the attribute provided as parameter"""
    col_values = [row[attr].item() for row in data]     
    unique_values = list(set(col_values))              
    labels = [row[-1].item() for row in data]                    
    total_len = len(col_values)
    avg_info = 0
    for val in unique_values:
        count = col_values.count(val)          
        prob = torch.tensor(count / total_len)  
        subset_attr = []
        subset_labels = []
        for i in range(total_len):
            if col_values[i] == val:
                subset_attr.append(val)
                subset_labels.append(labels[i])   
        subset_tensor = torch.cat((
            torch.tensor(subset_attr).unsqueeze(1),
            torch.tensor(subset_labels).unsqueeze(1)
        ), dim=1)     
        subset_entropy = get_entropy_of_dataset(subset_tensor)
        if not torch.isnan(torch.tensor(subset_entropy)):
            avg_info += (prob * subset_entropy).item()     
    return avg_info


def get_information_gain(data: torch.Tensor, attr: int):
    """Return Information Gain of the attribute provided as parameter"""
    return torch.round(
        torch.tensor(get_entropy_of_dataset(data)) - get_avg_info_of_attribute(data, attr),
        decimals=4
    ).item() 


def get_selected_attribute(data: torch.Tensor):
    """
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as an integer representing attribute number of selected attribute

    example : ({0: 0.123, 1: 0.768, 2: 1.23} , 2)
    """
    ig_scores = {}
    for i in range(len(data[0]) - 1):
        ig_scores[i] = get_information_gain(data, i)           
    max_gain = max(ig_scores.values())
    for i in ig_scores.keys():
        if ig_scores[i] == max_gain:
            return (ig_scores, int(i))                         
    return ({}, -1)
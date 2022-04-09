import argparse
import pandas as pd
import random

def build_balance_data(args):
    data = pd.read_csv(args.data_path)

    labels = data["utt_spk_int_labels"].values
    name = data["speaker_name"].values
    paths = data["utt_paths"].values
    durations = data["durations"].values
    
    #将各列数据都依label为索引值，建立字典方便使用label值来查找
    dict_name = {}
    dict_paths = {}
    dict_durations = {}
    for idx, label in enumerate(labels):
        if label not in dict_paths:
            dict_name[label] = name[idx]
            dict_paths[label] = []
            dict_durations[label] = []
        if abs(durations[idx] - 9) < 3:    #筛选语音长度，保证单条语音的长度不至于过大，也趋近于平均值
            dict_paths[label].append(paths[idx])
            dict_durations[label].append(durations[idx])


    #产生随机的说话人（args.num_spk不同个labels）,保存到列表random_num_spk
    candi_spk = []
    for label in range(max(labels) + 1):
        if args.utt_per_spk <= len(dict_paths[label]):  #筛选候选集合，保证长度足够可选
            candi_spk.append(label)
    random_num_spk = random.sample(candi_spk, args.num_spk)


    result_name = []
    result_paths = []
    result_durations = []
    result_labels = []
    for label in random_num_spk:            #dict_name[label]  dict_paths[label]  label  dict_durations[label]
        #对于每一个随机选出来的spk(label)，下面再随机选出utt_per_spk条不同的语音下标，保存到列表random_utt_per_spk
        candi_utt = [i for i in range(len(dict_paths[label]))]
        random_utt_per_spk = random.sample(candi_utt, args.utt_per_spk)
        #保存结果
        result_labels.extend([label] * args.utt_per_spk)
        for idx in random_utt_per_spk:  
            result_name.append(dict_name[label])
            result_paths.append(dict_paths[label][idx])
            result_durations.append(dict_durations[label][idx])

    table = {}
    for idx, label in enumerate(set(result_labels)):
        table[label] = idx

    labels = []
    for label in result_labels:
        labels.append(table[label])

    #写到csv文件
    dic = {'speaker_name': result_name, 'utt_paths': result_paths, 'utt_spk_int_labels': labels, 'durations': result_durations}    
    df = pd.DataFrame(dic)
    df.to_csv(args.save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/train.csv")
    parser.add_argument('--save_path', type=str, default="balance.csv")
    parser.add_argument('--num_spk', type=int, default=1211)
    parser.add_argument('--utt_per_spk', type=int, default=122)
    args = parser.parse_args()

    build_balance_data(args)

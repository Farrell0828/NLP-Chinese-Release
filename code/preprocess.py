import pandas as pd
from collections import OrderedDict
import json
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-tc-dirpath',
        default='../tcdata/nlp_round1_data/',
        help='The folder path to tianchi data.'
    )
    parser.add_argument(
        '--input-additional-dirpath',
        default='../user_data/additional_data/',
        help='The folder path to additional data.'
    )
    parser.add_argument(
        '--output-dirpath',
        default='../user_data/repreprocessed_data/',
        help='The folder path to output preprocessed data.'
    )
    return parser.parse_args()


def strQ2B(ustring):
    """
    全角转半角
    :param ustring: string with encoding utf8
    :return: string with encoding utf8
    """
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def same_origin(origin_tnews_path, target_tnews_path):
    """
    将额外数据转换成与官网提供数据格式一致的
    :param origin_tnews_path: 额外数据路径
           target_tnews_path: 目标数据路径
    """
    lines = open(origin_tnews_path, encoding='utf-8').readlines()
    data = []
    i = 0
    index = []
    clas = []
    dic = OrderedDict()

    for line in lines:
        line = line.split('_!_')
        index.append(i)
        if '\t' in line[3]:
            temp = ''
            for l in line[3].split('\t'):
                temp += l
            line[3] = temp
        data.append(strQ2B(line[3]))
        clas.append(line[1])
        i += 1

    dic['index'] = index
    dic['data'] = data
    dic['clas'] = clas
    f = pd.DataFrame.from_dict(dic)
    f.to_csv(target_tnews_path, index=False, header=['id', 'sentence', 'label'], encoding="utf-8")


def duplicated_recrate(lists, save_path, names):
    """
    [a, dev, train, other, all]
    :param a: tnews_test_a路径
           dev: tnews_dev路径
           train: tnews_train路径
           other: tnews_other路径
           all: tnews_all路径
    生成各文件去重后结果，保存csv文件
    """
    new_list = []
    new = []
    for i in lists:
        temp = i.drop_duplicates(subset=['sentence'], keep='first')
        new_list.append(temp)
    first = new_list[0]
    for index, i in enumerate(new_list):
        if index == 0:
            new.append(i.reset_index(drop = True))
            continue
        second = pd.concat([first, i])
        second = second.drop_duplicates(subset=['sentence'], keep='first')
        # new.append(second.sub(first).reset_index)
        temp_2 = second.iloc[len(first):].reset_index(drop = True)
        new.append(temp_2)
        first = second
    new.append(first)

    #生成各类去重文件
    for i, n in zip(new, names):
        i['id'] = list(range(len(i)))
        # if n == 'tnews_a.csv':
        #     i = i.drop(['label'], axis=1)
        #     i.to_csv(save_path + n, index=False, header=['id', 'sentence'], encoding="utf-8")
        # else:
        if n != 'tnews_a.csv':
            i['label'] = i['label'].astype(int)
            i.to_csv(save_path + n, index=False, encoding="utf-8")


def read_path_csv(list_path):
    """
    读取csv文件
    :param list_path: 需要读取的csv文件的路径列表
    :return data: 读取到的csv文件, [[],[]...]
    """
    data = []
    for index, path in enumerate(list_path):
        if index == 0:
            data_temp = pd.read_csv(path, sep='\\t', names=['id', 'sentence', 'label'], encoding='utf-8', engine='python')
        else:
            data_temp = pd.read_csv(path, sep=',', encoding='utf-8')
        data.append(data_temp)
    return data


def split_dev_train(path, path_dev, path_train):
    """
    将tnews_train1128切分成dev和train
    :param path: tnews_train1128路径
    """
    data = pd.read_csv(path, sep='\t', names=['id', 'sentence', 'label'], encoding='utf-8')
    data_dev = data[53360:]
    data_train = data[:53360]
    data_dev.to_csv(path_dev, index=False, header=['id', 'sentence', 'label'],
                    encoding="utf-8")
    data_train.to_csv(path_train, index=False, header=['id', 'sentence', 'label'], encoding="utf-8")


def create_tnews_dev_origin(input_additional_dirpath, output_dirpath):
    data = []
    with open(os.path.join(input_additional_dirpath, 'tnews_public/dev.json'), mode='r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    tnews_json_dev_df = pd.DataFrame(data)
    tnews_json_dev_df.index.name = 'id'
    tnews_json_dev_df['sentence'] = tnews_json_dev_df['sentence'].apply(strQ2B)
    tnews_json_dev_df[['sentence', 'label']].to_csv(os.path.join(output_dirpath, 'tnews_dev_origin.csv'), encoding='utf-8')



def create_tnews_train_origin(input_additional_dirpath, output_dirpath):
    data = []
    with open(os.path.join(input_additional_dirpath, 'tnews_public/train.json'), mode='r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    tnews_json_dev_df = pd.DataFrame(data)
    tnews_json_dev_df.index.name = 'id'
    tnews_json_dev_df['sentence'] = tnews_json_dev_df['sentence'].apply(strQ2B)
    tnews_json_dev_df[['sentence', 'label']].to_csv(os.path.join(output_dirpath, 'tnews_train_origin.csv'), encoding = 'utf-8')



def tnews_data_pre(input_tc_dirpath, input_additional_dirpath, output_dirpath):
    create_tnews_dev_origin(input_additional_dirpath, output_dirpath)
    create_tnews_train_origin(input_additional_dirpath, output_dirpath)

    origin_tnews_path = os.path.join(input_additional_dirpath, 'toutiao_cat_data.txt')
    target_tnews_path = os.path.join(output_dirpath, 'TNEWS_train_new_banjiao.csv')

    origin_tnews_a = os.path.join(input_tc_dirpath, 'TNEWS_a.csv')
    origin_tnews_dev = os.path.join(output_dirpath, 'tnews_dev_origin.csv')
    origin_tnews_train = os.path.join(output_dirpath, 'tnews_train_origin.csv')
    origin_tnews_other_banjiao = os.path.join(output_dirpath, 'TNEWS_train_new_banjiao.csv')

    lists = [origin_tnews_a, origin_tnews_dev, origin_tnews_train, origin_tnews_other_banjiao]
    # save_path = 'data/duplicate_tnews/'
    result_names = ['tnews_a.csv', 'tnews_dev.csv', 'tnews_train.csv', 'tnews_other.csv']

    same_origin(origin_tnews_path, target_tnews_path)
    # split_dev_train(tnews1128_path, origin_tnews_dev, origin_tnews_train)
    duplicated_recrate(read_path_csv(lists), output_dirpath, result_names)

    tnews_dev = os.path.join(output_dirpath, 'tnews_dev.csv')
    tnews_train = os.path.join(output_dirpath, 'tnews_train.csv')
    tnews_trainval_t = os.path.join(output_dirpath, 'tnews_trainval_t.csv')

    tnews_train_df = pd.read_csv(tnews_train, index_col = 0)
    tnews_val_df = pd.read_csv(tnews_dev, index_col = 0)
    tnews_trainval_df = pd.concat([tnews_train_df, tnews_val_df], axis = 0)
    tnews_trainval_df = tnews_trainval_df.reset_index(drop=True)
    tnews_trainval_df.index.name = 'id'
    tnews_trainval_df.to_csv(tnews_trainval_t, sep = '\t', encoding = 'utf-8', header = False)




'''*************************************    clue    ******************************************'''

def read_json(origin_file_path,write_file_path):
    data = []
    with open(origin_file_path, mode='r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    cmnli_json_train_df = pd.DataFrame(data)
    cmnli_json_train_df = cmnli_json_train_df[cmnli_json_train_df['label'] != '-']
    cmnli_json_train_df = cmnli_json_train_df.reset_index(drop=True)
    cmnli_json_train_df.index.name = 'id'
    cmnli_json_train_df['sentence1'] = cmnli_json_train_df['sentence1'].apply(strQ2B)
    cmnli_json_train_df['sentence2'] = cmnli_json_train_df['sentence2'].apply(strQ2B)
    cmnli_json_train_df[['sentence1', 'sentence2', 'label']].to_csv(write_file_path, encoding='utf-8')


def get_ocnli_trainval(input_tc_dirpath, input_additional_dirpath, output_dirpath):
    cmnli_train = os.path.join(output_dirpath, 'cmnli_train.csv')
    cmnli_dev = os.path.join(output_dirpath, 'cmnli_dev.csv')
    ocnli_train = os.path.join(output_dirpath, 'ocnli_train.csv')
    ocnli_dev = os.path.join(output_dirpath, 'ocnli_dev.csv')
    cmnli_trainval = os.path.join(output_dirpath, 'cmnli_trainval.csv')
    ocnli_trainval = os.path.join(output_dirpath, 'ocnli_trainval.csv')

    read_json(os.path.join(input_additional_dirpath, 'cmnli_public/train.json'), cmnli_train)
    read_json(os.path.join(input_additional_dirpath, 'cmnli_public/dev.json'), cmnli_dev)
    read_json(os.path.join(input_additional_dirpath, 'ocnli_public/train.50k.json'), ocnli_train)
    read_json(os.path.join(input_additional_dirpath, 'ocnli_public/dev.json'), ocnli_dev)

    # 把cmnli train和dev合并得到cmnli_trainval
    cmnli_train_df = pd.read_csv(cmnli_train, index_col=0)
    cmnli_dev_df = pd.read_csv(cmnli_dev, index_col=0)
    cmnli_trainval_df = pd.concat([cmnli_train_df, cmnli_dev_df], axis=0)
    cmnli_trainval_df = cmnli_trainval_df.reset_index(drop=True)
    cmnli_trainval_df.index.name = 'id'
    cmnli_trainval_df.to_csv(cmnli_trainval, encoding='utf-8')

    # 把ocnli train和dev合并得到ocnli_trainval
    df1 = pd.read_csv(ocnli_train, index_col=0)
    df2 = pd.read_csv(ocnli_dev, index_col=0)
    df = pd.concat([df1, df2], axis=0)
    df = df.reset_index(drop=True)
    df.index.name = 'id'
    df.to_csv(ocnli_trainval, encoding='utf-8')




'''*************************************    emotion    ******************************************'''
from transformers import AutoTokenizer
import re
import os
import emojiswitch


def replace_longSen(input_tc_dirpath, input_additional_dirpath, output_dirpath):
    file_selinux = os.path.join(input_tc_dirpath, 'OCEMOTION_train1128.csv')
    temp_file_selinux = file_selinux + '_clean' + '.temp'
    re_sub_list = ''
    manyWord = ['?', '*', '.', ')', '(', '^']

    with open(file_selinux, mode='r', encoding='utf-8') as fr, open(temp_file_selinux, mode='w', encoding='utf-8') as fw:
        for line in fr:
            re_sub_list = line
            for m in manyWord:
                b = re.search('(.)\\1{6,}', re_sub_list)
                if b != None:
                    b = b.group(0)
                    if b[0] != '?' and b[0] != ')' and b[0] != '*' and b[0] != '.' and b[0] != '^':
                        c = b[:6]
                        re_sub_list = re.sub(str(b), str(c), re_sub_list)
                    elif b[0] == m:
                        c = b[:6]
                        b = b.replace(m, '\\' + m)
                        re_sub_list = re.sub(b, c, re_sub_list)
            iter = re.finditer('(\[[\u4e00-\u9fa5]+\]){1}\\1+', re_sub_list)
            a = []
            for i in iter:
                if len(str(i)) > 96:
                    j = str(i).find('[')
                    a.append(str(i)[j:j + 10])
                else:
                    a.append(i.group())
            if len(a) > 0:
                # b = ''
                for i in a:
                    # b += i
                    # print(i)
                    c = i[:4]
                    i = i.replace('[', '\[')
                    i = i.replace(']', '\]')
                    re_sub_list = re.sub(i, c, re_sub_list)
                    # print(re_sub_list)
            re_sub_list = re.sub('[A-Za-z]+', '', re_sub_list[:-20]) + re_sub_list[-20:]
            fw.writelines(re_sub_list)
    # os.remove(file_selinux)
    if os.path.exists(os.path.join(output_dirpath, 'OCEMOTION_train1128_clean.csv')):
        os.remove(os.path.join(output_dirpath, 'OCEMOTION_train1128_clean.csv'))
    os.rename(temp_file_selinux, os.path.join(output_dirpath, 'OCEMOTION_train1128_clean.csv'))


def selinux_config(origin_line,update_line, output_dirpath):
    """
    关闭SELINUX
    修改文件内容
    :return:
    """
    file_selinux = os.path.join(output_dirpath, 'OCEMOTION_train1128_clean.csv')
    temp_file_selinux = file_selinux + '_clean' + '.temp'

    with open(file_selinux, mode='r', encoding='utf-8') as fr, open(temp_file_selinux, mode='w', encoding='utf-8') as fw:
        for line in fr:
            re_sub_list = re.sub(origin_line, update_line, line)  # 这里用re.sub进行替换后放入 re_sub_list中
            fw.writelines(re_sub_list)  # 将列表中的每一行进行写入。writelines是将序列对象中的每一行进行写入。
    os.remove(file_selinux)
    os.rename(temp_file_selinux, file_selinux)


def emotion_data_clean(input_tc_dirpath, input_additional_dirpath, output_dirpath):
    # 获取表情，dict
    emotion_df = []
    with open(os.path.join(input_tc_dirpath, 'OCEMOTION_train1128.csv'), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            for word in line:
                emotion_df.append(word)

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    vocab = tokenizer.vocab
    oov = {}
    for text in emotion_df:
        for c in text:
            if c not in vocab:
                if c in oov:
                    oov[c] += 1
                else:
                    oov[c] = 1

    # 去重字符串
    replace_longSen(input_tc_dirpath, input_additional_dirpath, output_dirpath)

    # 表情替换文本
    for k, v in tqdm(oov.items()):
        # selinux_config(k,emojiswitch.demojize(str(k),delimiters=("__","__"),lang='zh'))
        selinux_config(k, emojiswitch.demojize(str(k), lang='zh'), output_dirpath)


def test_convert(input_tc_dirpath, output_dirpath, test_file_path, write_file_path):
    test = os.path.join(input_tc_dirpath + test_file_path)
    data = pd.read_csv(test, sep='\t', names=['id', 'sentence', 'label'], encoding='utf-8')
    data['sentence'] = data['sentence'].apply(strQ2B)
    data.to_csv(os.path.join(output_dirpath + write_file_path), header=None, sep='\t', encoding='utf-8', index=False)


'''*************************************    all    ******************************************'''

def all_data_pre(input_tc_dirpath, input_additional_dirpath, output_dirpath):
    os.makedirs(output_dirpath, exist_ok=True)

    # ocnli
    print("ocnli trainval processing...")
    get_ocnli_trainval(input_tc_dirpath, input_additional_dirpath, output_dirpath)
    print("ocnli trainval finished !!!")

    # tnews
    print("tnews trainval processing...")
    tnews_data_pre(input_tc_dirpath, input_additional_dirpath, output_dirpath)
    print("tnews trainval finished !!!")

    # emotion
    print("emoction trainval processing...")
    emotion_data_clean(input_tc_dirpath, input_additional_dirpath, output_dirpath)
    print("emoction trainval finished !!!")

    #test_B
    print("test_B  processing...")
    test_convert(input_tc_dirpath, output_dirpath, 'ocemotion_test_B.csv', 'ocemotion_test_B_clean.csv')
    test_convert(input_tc_dirpath, output_dirpath, 'tnews_test_B.csv', 'tnews_test_B_clean.csv')
    print("test_B finished !!!")


def main():
    args = get_args()
    all_data_pre(args.input_tc_dirpath, args.input_additional_dirpath, args.output_dirpath)

if __name__ == '__main__':
    main()

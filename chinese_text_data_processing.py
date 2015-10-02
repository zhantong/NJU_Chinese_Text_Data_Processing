"""
Task Description:
1.First, please do the tokenization job to get the words from the Chinese sentences. You may use some existing tools like jieba for python, or IKAnalyzer for java, to help you accomplish the tokenization job.
2.Second, please delete the stop words. We provide a Chinese stop words list for your reference, and you can download it.
3.Third, please extract TF-IDF (Term Frequency–Inverse Document Frequency) features from the raw text data on the basis of step 1 and step 2 above. You may learn or review TF-IDF here. Please implement extracting TF-IDF features by yourself, and Do NOT invoke other existing codes or tools.
4.Four, please generate 10 new txt files which only contain TF-IDF features for each class, each line in each new txt file represents the TF-IDF feature of an instance/example in the corresponding class. Please name the 10 new txt files as same as the original 10 txt files that we provide.
--------------------------------
Python3.5 x64；
需要第三方库jieba支持。
stop words文件在同级目录下，文件名为Chinese-stop-words.txt；
需要进行分析的文件在同级lily文件夹下。
代码遍历lily文件夹下的所有文件，对每个文件首先分词、然后去除stop words，然后计算TF-IDF、最后根据TF-IDF值写入文件到同级result文件夹下。
"""
import jieba
import math
import os


class Process():

    def __init__(self):
        """初始化
        初始化声明需要用到的变量；
        判断用于保存结果的result文件夹是否存在，不存在则创建。
        """
        self.stop_words = set()  # 保存stop words
        self.glo_count = {}  # 保存所有文件中每个词语出现的行数（区别于次数）
        self.result = []  # 保存最终结果
        self.line_all = 0  # 保存所有文件总行数
        self.get_stop_words()  # 初始化时即读取stop words
        if not os.path.exists('result'):  # 保存结果文件的目录不存在则创建
            os.mkdir('result')

    def get_stop_words(self):
        """读取stop words
        从文件中读取stop words，保存到set类型的stop_words变量。
        """
        with open('Chinese-stop-words.txt', 'r', encoding='gbk') as f:
            for line in f:
                self.stop_words.add(line.strip())

    def tokenization(self, file_name):
        """分词并整理，消除stop words，计算词语出现次数
        将jieba分词得到每个单词加入到dict，以单词为key，value为dict
        result如[{{'a':{'count':1},...},'count_all':50},...]
        self.result还会记录文件名，为生成结果文件提供便利
        此外还会记录每行text的消除stop words后的词语总数，临时变量count
        如果某行出现某词，则在glo_count中对应加1，以计算词语出现总行数
        """
        res = []  # 每个文件临时结果
        with open('lily/' + file_name, 'r', encoding='utf-8') as f:
            for line in f:
                self.line_all += 1  # 总行数
                dic = {}  # 记录每一行的词语
                count = 0  # 每行中词语总数
                s = jieba.cut(line.strip())
                for item in s:
                    item = item.strip()
                    if not item:  # 忽略空字符串
                        continue
                    if not item in self.stop_words:  # 消除stop words
                        if not item in dic:  # 词语不在dic中则首先创建
                            dic[item] = {
                                'count': 0
                            }
                            if not item in self.glo_count:  # 不在glo_count中则首先创建
                                self.glo_count[item] = 0
                            self.glo_count[item] += 1  # 更新词语出现行数，只在此行中第一次出现时增加
                        dic[item]['count'] += 1  # 更新词语出现次数
                        count += 1  # 更新每行词语总数
                res.append({
                    'result': dic,
                    'count_all': count
                })
        self.result.append({
            'content': res,
            'file_name': file_name
        })

    def cal(self):
        """计算tf-idf
        通过分词得到的每行中各词语出现次数，
        每行词语总数，
        每个词语出现行数，
        总行数，
        可以计算得到tf-idf。
        """
        for f in self.result:
            for line in f['content']:
                tokens = line['result']
                count_all = line['count_all']  # 每行词语总数
                for token, values in tokens.items():
                    values['tf'] = values['count'] / count_all  # 计算tf
                    values['idf'] = 1 + math.log(
                        self.line_all / self.glo_count[token])  # 计算idf，加1避免idf为0
                    values['tf-idf'] = values['tf'] * values['idf']  # 计算tf-idf

    def write(self):
        """形成td-idf值矩阵，将结果写入文件
        """
        all_tokens = sorted(self.glo_count)  # 避免每次遍历时dict随机排序
        for doc in self.result:
            with open('result/' + doc['file_name'], 'w', encoding='utf-8') as f:
                for line in doc['content']:
                    for token in all_tokens:  # 以全部分词序列
                        if token in line['result']:
                            f.write('%.4f\t' % line['result'][token]['tf-idf'])
                        else:
                            f.write('0\t')
                    f.write('\n')

    def start(self):
        """多文件完整运行
        遍历lily文件夹下所有文件。
        """
        if not os.path.isdir('lily'):
            print('sample floder not found')
            return
        print('遍历文件...')
        for file_name in os.listdir('lily'):  # 遍历lily下的文件
            self.tokenization(file_name)
        print('计算TF-IDF...')
        self.cal()
        print('写入文件...')
        self.write()
        print('完成！')

if __name__ == '__main__':
    s = Process()
    s.start()

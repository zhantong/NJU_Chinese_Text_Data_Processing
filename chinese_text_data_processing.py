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
代码遍历lily文件夹下的所有文件，对每个文件首先分词、然后去除stop words、然后计算TF-IDF、最后根据TF-IDF排序并写入文件到同级result文件夹下。
"""
import jieba
import math
import os


class Process():

    def __init__(self):
        """初始化
        初始化声明需要用到的变量；
        判断用于保存结果的result是否存在，不存在则创建。
        """
        self.stop_words = set()  # 保存stop words
        self.glo_count = {}  # 保存文件中每个词语出现的行数（区别于次数）
        self.result = []  # 保存最终结果
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
        self.result如[{{'a':{'count':1},...},'count_all':50},...]
        此外还会记录每行text的词语总数，临时变量count
        以及每个词语共出现在多少行text中，保存到glo_count
        """
        with open('lily/' + file_name, 'r', encoding='utf-8') as f:
            for line in f:
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
                            self.glo_count[item] += 1  # 更新词语出现行数
                        dic[item]['count'] += 1  # 更新词语出现次数
                        count += 1  # 更新每行词语总数
                self.result.append({
                    'result': dic,
                    'count_all': count
                })

    def cal(self):
        """计算tf-idf
        通过分词得到的每行text中各词语出现次数，
        每行text词语总数，
        每个词语出现text行数，
        总行数，
        可以计算得到tf-idf。
        """
        lines = len(self.result)  # 总行数
        for line in self.result:
            tokens = line['result']
            count_all = line['count_all']  # 每行词语总数
            for token, values in tokens.items():
                values['tf'] = values['count'] / count_all  # 计算tf
                values['idf'] = math.log10(
                    lines / self.glo_count[token])  # 计算idf
                values['tf-idf'] = values['tf'] * values['idf']  # 计算tf-idf

    def sort_and_write(self, file_name):
        """以td-idf值由大到小排序，将结果写入文件
        """
        all_tokens = sorted(self.glo_count)
        with open('result/' + file_name, 'w', encoding='utf-8') as f:
            for line in self.result:
                for token in all_tokens:
                    if token in line['result']:
                        f.write('%.4f\t' % line['result'][token]['tf-idf'])
                    else:
                        f.write('0\t')
                f.write('\n')

    def start_once(self, file_name):
        """单次完整运行
        只处理一个文件，
        每次开始处理前初始化变量。
        """
        self.glo_count = {}  # 重新初始化变量
        self.result = []
        self.tokenization(file_name)
        self.cal()
        self.sort_and_write(file_name)

    def start(self):
        """多文件完整运行
        遍历lily文件夹下所有文件。
        """
        if not os.path.isdir('lily'):
            print('sample floder not found')
            return
        for file_name in os.listdir('lily'):  # 遍历lily下的文件
            self.start_once(file_name)

if __name__ == '__main__':
    s = Process()
    s.start()

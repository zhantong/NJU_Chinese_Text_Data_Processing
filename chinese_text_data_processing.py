import jieba
import math

stop_words=set()
def get_stop_words():
	global stop_words
	with open('Chinese-stop-words.txt','r',encoding='gbk') as f:
		for line in f:
			stop_words.add(line.strip())

result=[]
glo_count={}
def tokenization():
	global result
	global glo_count
	with open('lily/Basketball.txt','r',encoding='utf-8') as f:
		for line in f:
			dic={}
			count=0
			s=jieba.cut(line.strip())
			for item in s:
				if not item in stop_words:
					if not item in dic:
						dic[item]={
							'count':0
						}
						if not item in glo_count:
							glo_count[item]=0
						glo_count[item]+=1
					dic[item]['count']+=1
					count+=1
			result.append({
				'result':dic,
				'count_all':count
				})

def cal():
	lines=len(result)
	for line in result:
		tokens=line['result']
		count_all=line['count_all']
		for token,values in tokens.items():
			values['tf']=values['count']/count_all
			values['idf']=math.log10(lines/glo_count[token])
			values['tf-idf']=values['tf']*values['idf']
			values['test']=glo_count[token]

def sort():
	for line in result:
		print('--------------------------------------')
		for key in sorted(line['result'],key=lambda k:line['result'][k]['tf-idf'],reverse=True):
			print(key,line['result'][key])

if __name__=='__main__':
	get_stop_words()
	tokenization()
	cal()
	sort()
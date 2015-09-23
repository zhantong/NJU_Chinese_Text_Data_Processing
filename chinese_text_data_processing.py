import jieba
import math


class Process():
	def __init__(self):
		self.stop_words=set()
		self.glo_count={}
		self.result=[]
		self.get_stop_words()
	def get_stop_words(self):
		global stop_words
		with open('Chinese-stop-words.txt','r',encoding='gbk') as f:
			for line in f:
				self.stop_words.add(line.strip())

	def tokenization(self,file_name):
		with open('lily/'+file_name,'r',encoding='utf-8') as f:
			for line in f:
				dic={}
				count=0
				s=jieba.cut(line.strip())
				for item in s:
					item=item.strip()
					if not item:
						continue
					if not item in self.stop_words:
						if not item in dic:
							dic[item]={
								'count':0
							}
							if not item in self.glo_count:
								self.glo_count[item]=0
							self.glo_count[item]+=1
						dic[item]['count']+=1
						count+=1
				self.result.append({
					'result':dic,
					'count_all':count
					})

	def cal(self):
		lines=len(self.result)
		for line in self.result:
			tokens=line['result']
			count_all=line['count_all']
			for token,values in tokens.items():
				values['tf']=values['count']/count_all
				values['idf']=math.log10(lines/self.glo_count[token])
				values['tf-idf']=values['tf']*values['idf']

	def sort_and_write(self,file_name):
		with open('result/'+file_name,'w',encoding='utf-8') as f:
			for line in self.result:
				for key in sorted(line['result'],key=lambda k:line['result'][k]['tf-idf'],reverse=True):
					f.write('%s=%.4f\t'%(key,line['result'][key]['tf-idf']))
				f.write('\n')
	def start_once(self,file_name):
		self.glo_count={}
		self.result=[]
		self.tokenization(file_name)
		self.cal()
		self.sort_and_write(file_name)

if __name__=='__main__':
	s=Process()
	s.start_once('Basketball.txt')
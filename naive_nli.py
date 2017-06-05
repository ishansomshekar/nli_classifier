import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

import data

lang_dict = {
    'HIN' : 0,
    'ARA' : 1,
    'JPN' : 2,
    'SPA' : 3,
    'TUR' : 4,
    'GER' : 5,
    'TEL' : 6,
    'KOR' : 7,
    'ITA' : 8,
    'CHI' : 9,
    'FRE' : 10
}

class NaiveGRU:
	def __init__(self):
		pass

	def _create_placeholders(self):
		pass

	def build_graph(self):
		pass

def train(model, batch):
	pass

def eval(model, batch):
	pass

def main():
	model = NaiveGRU()
	dataset = 'essay'
	mode = 'tokenized'
	batch = data.batch_gen()
	model.build_graph()
	train(model)
	eval(model)

if __name__ == '__main__':
	main()


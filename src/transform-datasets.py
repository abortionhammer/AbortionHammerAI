import os
import random

def export_characters_diversity(dataset_file_path, output_file_path):
  with open(dataset_file_path, 'r', encoding="utf-8") as fr:
    data = fr.read().replace('\n', '')
    chars = ''.join(set(data))
    with open(output_file_path, 'w', encoding="utf-8") as fw:
      fw.write(chars)
	  
	  
def regenerate_ids(dataset_file_path, output_file_path):
  with open(dataset_file_path, 'r', encoding="utf-8") as fr:
    lines = fr.readlines()
  with open(output_file_path, 'w', encoding="utf-8") as fw:
    i = 1
    c = 1
    for line in lines:
      if c > 1:
        fields = line.split('\t')
        new_line = "\t".join(["{:05d}".format(i), fields[1], fields[2], fields[3]])
        fw.write(new_line)
        i = i + 1
      else:
        fw.write(line)
      c = c + 1
	  
	  
def shuffle(dataset_file_path, output_file_path):
  with open(dataset_file_path, 'r', encoding="utf-8") as fr:
    lines = fr.readlines()
  random.shuffle(lines)
  with open(output_file_path, 'w', encoding="utf-8") as fw:
    for line in lines:
      fw.write(line)	
	
	
#export_characters_diversity("../datasets/tweets/semeval-favor-against-none.txt", "../datasets/tweets/characters.txt")
#regenerate_ids("../datasets/tweets/semeval-favor-against-none.txt", "../datasets/tweets/semeval-favor-against-none-new-ids.txt")
#shuffle("../datasets/tweets/semeval-favor-against-none-new-ids.txt", "../datasets/tweets/semeval-favor-against-none-shuffled.txt")
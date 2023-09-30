import os

prefix = '/Users/lilee/Desktop/code/autotuning/data/SIDD_CROP/'

for i in range(1, 2701):
    nz_path = os.path.join(prefix, 'NOISY', str(i) + '.PNG')
    red_path = os.path.join(prefix, 'RED', str(i) + '.PNG')
    gt_path = os.path.join(prefix, 'GT', str(i) + '.PNG')
    params_path = os.path.join(prefix, 'PARAMS', str(i) + '.txt')
    
    line = nz_path + ',' + red_path + ',' + params_path + '\n'
    with open('train.txt', 'a') as f:
        f.write(line)
        
        
for i in range(2701, 3001):
    nz_path = os.path.join(prefix, 'NOISY', str(i) + '.PNG')
    red_path = os.path.join(prefix, 'RED', str(i) + '.PNG')
    gt_path = os.path.join(prefix, 'GT', str(i) + '.PNG')
    params_path = os.path.join(prefix, 'PARAMS', str(i) + '.txt')
    
    line = nz_path + ',' + red_path + ',' + params_path + '\n'
    with open('val.txt', 'a') as f:
        f.write(line)

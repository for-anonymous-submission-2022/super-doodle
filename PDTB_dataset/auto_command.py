import os
import time

#print("*"*10+"all+EntRel"+"*"*10)
#start = time.time()
#os.system('python pdtb3_multi_preprocess.py '
#            '--data_dir PDTB_3.0/data '
#            '--output_dir EDiTS_datasets/all+EntRel '
#            '--which_dataset "all+EntRel" '
#            '--split L2_xval '
#            '--select_relations All '
#            '--create_sections Yes')
#end = time.time()
#print(f"---------- Time taken: {end - start:.5f} sec ----------")
#print("*"*10+"*"*10+"*"*10)
#print("*"*10+"*"*10+"*"*10)
#print("*"*10+"*"*10+"*"*10)
#
#print("*"*10+"exp-L2-14way"+"*"*10)
#start = time.time()
#os.system('python pdtb3_multi_preprocess.py '
#            '--data_dir PDTB_3.0/data '
#            '--output_dir EDiTS_datasets/exp-L2-14way '
#            '--which_dataset "exp-L2-14way" '
#            '--split L2_xval '
#            '--select_relations Explicit '
#            '--create_sections Yes')
#end = time.time()
#print(f"---------- Time taken: {end - start:.5f} sec ----------")
#print("*"*10+"*"*10+"*"*10)
#print("*"*10+"*"*10+"*"*10)
#print("*"*10+"*"*10+"*"*10)
#
#print("*"*10+"nexp-L2-14way+EntRel"+"*"*10)
#start = time.time()
#os.system('python pdtb3_multi_preprocess.py '
#            '--data_dir PDTB_3.0/data '
#            '--output_dir EDiTS_datasets/nexp-L2-14way+EntRel '
#            '--which_dataset "nexp-L2-14way+EntRel" '
#            '--split L2_xval '
#            '--select_relations Non-explicit '
#            '--create_sections Yes')
#end = time.time()
#print(f"---------- Time taken: {end - start:.5f} sec ----------")
#print("*"*10+"*"*10+"*"*10)
#print("*"*10+"*"*10+"*"*10)
#print("*"*10+"*"*10+"*"*10)
#
#print("*"*10+"all+EntRel+NotMat"+"*"*10)
#start = time.time()
#os.system('python pdtb3_multi_preprocess.py '
#            '--data_dir PDTB_3.0/data '
#            '--output_dir EDiTS_datasets/all+EntRel+NotMat '
#            '--which_dataset "all+EntRel+NotMat" '
#            '--split L2_xval '
#            '--select_relations All '
#            '--create_sections Yes')
#end = time.time()
#print(f"---------- Time taken: {end - start:.5f} sec ----------")
#print("*"*10+"*"*10+"*"*10)
#print("*"*10+"*"*10+"*"*10)
#print("*"*10+"*"*10+"*"*10)

print("*"*10+"exp-L2-9way+NotCon"+"*"*10)
start = time.time()
os.system('python pdtb3_multi_preprocess.py '
            '--data_dir PDTB_3.0/data '
            '--output_dir EDiTS_datasets/exp-L2-9way+NotCon '
            '--which_dataset "exp-L2-9way+NotCon" '
            '--split L2_xval '
            '--select_relations Explicit '
            '--create_sections Yes')
end = time.time()
print(f"---------- Time taken: {end - start:.5f} sec ----------")
print("*"*10+"*"*10+"*"*10)
print("*"*10+"*"*10+"*"*10)
print("*"*10+"*"*10+"*"*10)

print("*"*10+"nexp-L2-9way+EntRel+NotCon"+"*"*10)
start = time.time()
os.system('python pdtb3_multi_preprocess.py '
            '--data_dir PDTB_3.0/data '
            '--output_dir EDiTS_datasets/nexp_L2_9way+EntRel+NotCon '
            '--which_dataset "nexp-L2-9way+EntRel+NotCon" '
            '--split L2_xval '
            '--select_relations Non-explicit '
            '--create_sections Yes')
end = time.time()
print(f"---------- Time taken: {end - start:.5f} sec ----------")
print("*"*10+"*"*10+"*"*10)
print("*"*10+"*"*10+"*"*10)
print("*"*10+"*"*10+"*"*10)
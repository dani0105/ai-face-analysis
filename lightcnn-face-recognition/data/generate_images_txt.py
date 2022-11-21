import os

celeb_dir="lightcnn-face-recognition/data/input"
out_dir="lightcnn-face-recognition/data/input/images.txt"


with open(out_dir,'w') as F:
  for folder_class in os.listdir(celeb_dir):
    if(os.path.isfile(celeb_dir+"/"+folder_class)):
      continue
    for file in os.listdir(celeb_dir+"/"+folder_class):
      F.write(folder_class+"/"+file+" "+folder_class+"\n")

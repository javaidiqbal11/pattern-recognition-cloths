import os

# set the path 
path = "./pattern-recognition/train"
for (path, dirs, files) in os.walk(path):
    for dir in dirs:
        print(os.path.join(path, dir))
        os.system(f"mogrify -format jpg {os.path.join(path, dir)}/*.webp")
        os.system(f"rm {os.path.join(path, dir)}/*.webp")


path = "./pattern-recognition/test"
for (path, dirs, files) in os.walk(path):
    for dir in dirs:
        print(os.path.join(path, dir))
        os.system(f"mogrify -format jpg {os.path.join(path, dir)}/*.webp")
        os.system(f"rm {os.path.join(path, dir)}/*.webp")

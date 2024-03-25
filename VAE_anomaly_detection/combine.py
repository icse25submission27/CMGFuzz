import os
import shutil

src_dirs = ["./CMGValid", "./CMGID"]
dst_dir = "./CMGCleaned"
cnt=0
# 遍历源文件夹
for src_dir in src_dirs:
    for category in os.listdir(src_dir):
        src_path = os.path.join(src_dir, category)
        dst_path = os.path.join(dst_dir, category)

        # 如果目标路径不存在，则创建它
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        # 遍历源路径下的所有文件
        for file_name in os.listdir(src_path):
            # 构造完整的文件路径
            full_file_name = os.path.join(src_path, file_name)
            if os.path.isfile(full_file_name):
                # 将文件复制到目标路径
                shutil.copy(full_file_name, dst_path+"/"+str(cnt)+".jpg")
                cnt+=1
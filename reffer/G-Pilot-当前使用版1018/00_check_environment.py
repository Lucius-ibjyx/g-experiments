import importlib.util,shutil,sys
ok=True
for name in ['numpy','pandas','openpyxl','matplotlib']:
 found=importlib.util.find_spec(name) is not None
 print(name, 'OK' if found else '缺失');ok &= found
for name in ['ffprobe','ffmpeg']:
 found=shutil.which(name);print(name,found or '缺失');ok &= bool(found)
print('环境可用' if ok else '请按使用说明安装依赖')
sys.exit(0 if ok else 1)

1. mdb文件生成
```text
data = {'coordinates': pred, 'writer_id': writer_id[i].item(),
  'character_id': character_id[i].item(), 'coords_gt': coord}
data_byte = pickle.dumps(data)
data_id = str(num_count).encode('utf-8')
test_cache[data_id] = data_byte

writeCache
```

2. img转pkl
```text
3. file_path = '.'
file_name = 'test.pkl'
imgs_path = ['../style_samples/1_binary.jpg', '../style_samples/2_binary.jpg', '../style_samples/3_binary.jpg']
write_pkl(file_path, file_name, imgs_path, 2)
```
额外的装饰网络，为SDT生成的均匀笔画的文字增加了笔画宽度和颜色?
how?

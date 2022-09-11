# 文：中文 LSP 服务的尝试

依赖安装： `pip install pygls pypinyin`

## 拼音输入法
使用方法：
```
# 下载本 repo 到本地，注意取名，以下用的是 wen, 如果用别的名字，后续操作保持一致即可
git clone git@github.com:metaescape/Wen.git ~/codes/wen
# 进入刚下载的仓库目录
cd ~/codes/wen
# 下载拼音输入法，注意名字为 completion （这个库有模型参数，比较大，网络不好可能需要点时间）
git clone git@github.com:whatbeg/GodTian_Pinyin.git completion
# 确保当前目录下有一个 completion/ 子目录
# 进入 completion 目录，执行以下命令打上补丁，目的是把原拼音项目里 python2 语法转成 python3
cd completion
git apply ../0001-for-py3.patch
```

你可以打开补丁看看，基本就是把 `print xxx` 改成 `print(xxx)`, 以及少量 py2 到 py3 接口变化


在系统路径下建立 ~/codes/wen/server.py 服务的软链接，名字需要是 wenls（确保 emacs 中能够直接执行 wenls 这个命令，这样 lsp-bridge 才能正确建立通信），比如：
```
ln -s ~/codes/wen/server.py ~/.local/bin/wenls
```

当然如果你是 sudo 用户，也可以把软链接建立在 /bin, /usr/bin, /usr/local/bin 这些系统路径下，如
```
ln -s ~/codes/wen/server.py /usr/bin/wenls
```

在 emacs org-mode 里执行 lsp-bridge-mode 启动 lsp 服务即可体验用补全进行拼音到汉字的输入转换

注意：以上输入法只是 demo 型的，词汇有限，有时候拼音切分不准也会导致部分词汇无法输出。

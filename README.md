# 文：中文 LSP 服务的尝试

当前使用方法：
下载本仓库，同时克隆 https://github.com/whatbeg/GodTian_Pinyin 到本目录中

将本目录中的 0001-for-py3.patch 移动到 GodTian_Pinyin 中并执行以下命令把补丁打上（目的是把项目里 python2 语法转成 python3 ），如果补丁失败，你可以打开 patch 文件手动修改（这就是一个 diff 文件，要修改的不超过 10 行）
```
git am --signoff < 0001-for-py3.patch
```

在 lsp-bridge 的 langserver 目录下新建 wen.json, 写入以下内容：

```json
{
  "name": "文",
  "languageId": "文",
  "command": ["wenls"],
  "settings": {}
}
```

lsp-bridge 的配置：
```
(use-package lsp-bridge
  :load-path "site-lisp/lsp-bridge"
  :hook ((java-mode org-mode) . lsp-bridge-mode)
  :config
  (add-to-list 'lsp-bridge-single-lang-server-mode-list
               '(org-mode . "wen"))
  )
```


使得在系统路径下能够找到 wenls 服务（确保 emacs 中能够直接执行 wenls 这个命令）
```
ln -s ~/codes/wen/server.py ~/.local/bin/wenls
```

import os
import json
import os.path as osp

CONFIG_DIR = osp.expanduser("~/.config/wen/")


class WenConfig:
    def __init__(self):
        self.setting = self.read_config()
        self.latex_table = self.read_latex_table()
        self.debug = self.setting.get("debug", False)
        self.completion = self.setting.get("completion", "gpt2")
        self.use_pinyin_initial = self.setting.get("use_pinyin_initial", False)
        self.model_path = self.setting.get("model_path", "aihijo/gpt2-zh-21k")
        self.python_path = self.setting.get("python_path", "/usr/bin/python3")

    def read_config(self):
        if not os.path.exists(CONFIG_DIR):
            os.mkdir(CONFIG_DIR)
        settings_path = os.path.join(CONFIG_DIR, "settings.json")
        if os.path.exists(settings_path):
            with open(settings_path, "r") as f:
                settings = json.load(f)
                return settings

    # def read_rime_vocab(settings):
    #     vocab_paths = settings["vocabularies"]
    #     vocabs = []
    #     for path in vocab_paths["rime"]:
    #         vocabs.extends(read_rime_vocab(path))
    #     return vocabs

    def read_latex_table(self):
        if "latex.table.path" not in self.setting:
            print("latex table file not set")
            return []
        path = osp.expanduser(self.setting["latex.table.path"])
        if not os.path.exists(path):
            print("latex table file not find")
            return []
        table = []
        with open(path) as f:
            for line in f.readlines():
                items = line.strip().split()
                if len(items) == 3:
                    table.append(items)
        return table

from dataloader import *
from PRLmodel import *
from tokenizer import *
import numpy as np
import itertools
import tqdm

DEFAULT_DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

class PRLEvaluator:
    def __init__(self, 
                 model_path, 
                 t1 = KBDPasswordTokenizer(), 
                 t2 = TransTokenizer(), 
                 batch_size=32, 
                 max_len=16, 
                 device=DEFAULT_DEVICE):
        self.t1 = t1 
        self.t2 = t2
        self.max_len = max_len
        self.model = torch.load(model_path).to(device)
        self.model.device = device
        self.device = device
        self.batch_size = batch_size
        self.model.set_mode("predict")
        self.model.eval()
    
    def predict(self, pwds):
        raise NotImplementedError()

class PRLGreedyEvaluator(PRLEvaluator):
    def __init__(self, 
                 model_path, 
                 t1=KBDPasswordTokenizer(), 
                 t2=TransTokenizer(), 
                 batch_size=32, 
                 max_len=16, 
                 device=DEFAULT_DEVICE):
        super().__init__(model_path, t1, t2, batch_size, max_len, device)
        self.model.set_mode("predict")

    def _predict(self, pwds):
        n = len(pwds)
        pwd_ids = self.t1(pwds, padding=False)
        pwd_ids = self.t1.padding(pwd_ids).to(self.device)
        digits = self.model(pwd_ids)

        probs = [sum(x[1] for x in d) for d in digits]
        edits = [[x[0] for x in d] for d in digits]
        return zip(pwds,[self.t2.decode(pwds[i], edits[i]) for i in range(n)], probs)
    

    def predict(self, pwds):
        n = len(pwds)
        combined = iter([])
        for i in range(0, n, self.batch_size):
            batch_x = pwds[i:i + self.batch_size]
            combined = itertools.chain.from_iterable([combined, self._predict(batch_x)])
        return combined

class PRLBeamSearchEvaluator(PRLEvaluator):
    def __init__(self, 
                 model_path, 
                 t1=KBDPasswordTokenizer(), 
                 t2=TransTokenizer(), 
                 batch_size=32, 
                 max_len=16, 
                 device=DEFAULT_DEVICE):
        super().__init__(model_path, t1, t2, batch_size, max_len, device)
        self.model.set_mode("beamsearch")


    def _predict_test(self, pwds,beamwidth=10, topk=2):
        n = len(pwds)
        pwd_ids = self.t1(pwds, padding=False)
        pwd_ids = self.t1.padding(pwd_ids).to(self.device)
        
        digits = self.model(pwd_ids,beamwidth=beamwidth, topk=topk)
        ans = []
        for i in range(n):
            items = []
            outputs = digits[i]
            for output in outputs:
                prob = output[1]
                edits = output[0]
                pwd = edits
                items.append((pwd, prob))
            ans.append(items)
        return zip(pwds,ans)
    

    def _predict(self, pwds,beamwidth=10, topk=2):
        n = len(pwds)
        pwd_ids = self.t1(pwds, padding=False)
        pwd_ids = self.t1.padding(pwd_ids).to(self.device)
        
        digits = self.model(pwd_ids,beamwidth=beamwidth, topk=topk)
        ans = []
        used_pwds = set()
        for i in range(n):
            items = []
            outputs = digits[i]
            for output in outputs:
                if len(items) >topk:
                    break
                prob = output[1]
                edits = output[0]
                pwd = self.t2.decode(pwds[i], edits)
                if pwd not in used_pwds:
                    items.append((pwd, prob))
                    used_pwds.add(pwd)
            ans.append(items)
        return zip(pwds,ans)
    
    def predict(self, pwds, beamwidth=10, topk=2):
        n = len(pwds)
        combined = iter([])
        for i in range(0, n, self.batch_size):
            batch_x = pwds[i:i + self.batch_size]
            combined = itertools.chain.from_iterable([combined, self._predict(batch_x, beamwidth=beamwidth, topk=topk)])
        return combined
    
    def predict_one(self, pwd, beamwidth=10, topk=2):
        return next(self._predict([pwd], beamwidth=beamwidth, topk=topk))

class FileEvaluator:
    def __init__(self, model,outputs):
        self.model = model
        self.csv_out = open(outputs, "w")
        
    def count_lines(self, file_path):
        line_count = 0
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    line_count += 1
        except FileNotFoundError:
            print("文件不存在：", file_path)
        except IOError:
            print("无法打开文件：", file_path)
        return line_count
    
    def parse_batch(self, inputs):
        src = [x[0] for x in inputs]
        tar = [x[1] for x in inputs]
        return src, tar

    def evaluate(self, path, beamwidth=150, topk=1000, batch_size=64):
        hit_count = 0
        pwds = []
        batch_size = batch_size
        with open(path, "r") as f:
            for line in f:
                line = line.strip("\r\n").split("\t")
                src = line[0]
                target = line[1]
                pwds.append((src, target))
        for i in tqdm.tqdm(range(0, len(pwds), batch_size)):
            inputs = pwds[i:i+batch_size]
            src, tar = self.parse_batch(inputs)
            outputs = list(self.model.predict(src, beamwidth, topk))
            for item in zip(inputs, outputs):
                hit = False
                (src, tar), (_, output) = item
                cnt = 0
                for pwd, prob in output:
                    if pwd == tar:
                        hit = True
                        hit_count += 1
                        self.csv_out.write(f"{src}\t{tar}\t{cnt}\t{prob}\n")
                        break
                    cnt += 1
                if not hit:
                    self.csv_out.write(f"{src}\t{tar}\t{-1}\t{0.0}\n")
        print(f">>> Guess Rate: {hit_count / len(pwds)}")

    def finish(self):
        self.csv_out.close()


def main():
    # model_load = "/disk/zsh/PasswordRL/model/testmodel.pt"
    model_load = "path_of_model"
    model = PRLBeamSearchEvaluator(model_load, device=DEFAULT_DEVICE)
    
    print(model.model)
    # evaluator = FileEvaluator(model,"0.1weight_v6")
    # path = "/disk/data/targuess/3_query/4iQ_10k.csv"
    evaluator = FileEvaluator(model,"output_path")
    path = "path_of_testfile"
    evaluator.evaluate(path)

    # pwds = [
    #     ("hello1", "lilcoach12345@yahoo.com"), 
    #     ("funtik44", "elena44.114@yandex.ru"),
    #     ("jebstone", "del7734@yahoo.com"),
    #     ("lerev1231", "verelius@gmail.com"), 
    #     ("a12345", "travel.m@hotmail.com"),
    #     ("hello1", ""), 
    #     ("funtik44", ""),
    #     ("jebstone", ""),
    #     ("lerev1231", ""), 
    #     ("a12345", "")
    # ]

    # test = [x[0] for x in pwds]


    # ans = model.predict(test, beamwidth=150, topk=100)

        
if __name__ == '__main__':
    main()
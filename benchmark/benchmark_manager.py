import importlib
# Base Class
class EvaluatorBase:
    def __init__(self) -> None:
        self.name = __name__
        self.log_base = {}
    def eval(self, pred, gt):
        pass
    def log(self, name, var):
        pass
    def get_log(self):
        return self.log_base
    

# get evaluator by its name
class EvaluatorFactory:
    def __init__(self) -> None:
        self.module_name = 'benchmark'
    def get_evaluator(self, eval_name: str)->EvaluatorBase:
        eval_module_name = f"{self.module_name}.evaluators.{eval_name}"
        eval_module = importlib.import_module(eval_module_name)
        eval_class = getattr(eval_module, eval_name)
        return eval_class

# BenchMark: a pipeline of evaluators
class Benchmark:
    def __init__(self, evaluator_list: list) -> None:
        print(f'Benchmakr get evaluator list: {evaluator_list}')
        self.eval_list = []
        self.bechmark_log = {}
        fac = EvaluatorFactory()
        for name in evaluator_list:
            evaluator_class = fac.get_evaluator(name)
            evaluator = evaluator_class()
            self.eval_list.append(evaluator)
        print("Preparation is finished...")
        
    def eval(self, pred, gt):
        for evaluator in self.eval_list:
            evaluator.eval(pred, gt)
            e_log = evaluator.get_log()
            e_name = evaluator.__class__.__name__
            self.bechmark_log[e_name] = e_log

    def get_log(self):
        return self.flatten_dict(self.bechmark_log)
    # @staticmethod
    def flatten_dict(self, d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}:{k}" if parent_key else k
            if isinstance(v, dict):
                # v, new_key, sep=sep).items()
                items.extend(self.flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    

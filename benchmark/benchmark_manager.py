import importlib
# Base Class
class EvaluatorBase:
    def __init__(self, output='stdout') -> None:
        pass
    def eval(self, pred, gt):
        pass

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
        fac = EvaluatorFactory()
        for name in evaluator_list:
            evaluator_class = fac.get_evaluator(name)
            evaluator = evaluator_class()
            self.eval_list.append(evaluator)
        print("Preparation is finished...")
        
    def eval(self, pred, gt):
        for evaluator in self.eval_list:
            evaluator.eval(pred, gt)

    

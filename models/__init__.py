import importlib
from .model_int_base import ModelIntBase
'''
Select the class of ModelInt according to model_int_name
'''
def ModelInt_Class_Select(model_int_name:str)->ModelIntBase:
    try:
        module_name = f"{__name__}.{model_int_name}.model_int"
        model_int_module = importlib.import_module(module_name)
        model_int_class = getattr(model_int_module, "ModelInt")
        return model_int_class
    except:
        raise ImportError(f"{model_int_name} is not found...")
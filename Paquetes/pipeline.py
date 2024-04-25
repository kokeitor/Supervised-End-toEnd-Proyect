
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,VarianceThreshold,f_classif
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from typing import Optional, Callable, List, Dict, Tuple,Sequence


from abc import ABCMeta, abstractmethod,abstractstaticmethod

class PipelineBaseCreator(metaclass=ABCMeta):
  

  @abstractstaticmethod
  def create_pipeline():
    """Create a Pipeline object"""
    
  @abstractmethod
  def add_steps():
    """set steps of the pipeline"""

class PipelineCreator(PipelineBaseCreator):
  
  __instance = None
  @staticmethod
  def create_pipeline(steps):
    if isinstance(steps, (list)):
      return Pipeline(steps = steps).steps.extend
    elif isinstance(steps, (tuple)):
      return Pipeline().steps.extend()
    else:
      raise ValueError("steps must be a list or tuple")
    
    return Pipeline(steps = steps)
  
  def __init__(self):
    self.steps = []
  def __rpr__(self):
    return "hola"
  def add_steps(self,steps):
    if isinstance(steps, (list)):
      return Pipeline().steps.extend
    elif isinstance(steps, (tuple)):
      return Pipeline().steps.extend()
    else:
      raise ValueError("steps must be a list or tuple")
    

  
  
  
def testing():
  steps = [("sgd",SGDRegressor())]

  pipe = PipelineCreator.create_pipeline(steps = steps)
  pipe2 = PipelineCreator()
  pipe2.add_steps(steps = steps)
  
  print()
  
if __name__ == "__main__":
  testing()
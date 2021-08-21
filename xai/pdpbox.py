from pdpbox import pdp, info_plots
import pandas as pd 
from typing import Any, List

class VizForPDPbox:

  def __doc__(self):
    '''
    package: pdpbox (pip install pdpbox)
    support: sckit-learn model only
    purpose: Visualize the amount of change in the model with respect to the amount of features
    '''

  def __init__(self):
    pass 

  def actual_plot(self, model: Any, x_train: pd.DataFrame, feature: str, show_percentile: bool=True):
    fig, axes, summary = info_plots.actual_plot(model=model,
                                                X=x_train,
                                                feature=feature,
                                                feature_name=feature, 
                                                show_percentile=show_percentile,
                                                predict_kwds={})
    
  def isolate_plot(self, model: Any, dataset: pd.DataFrame, feature: str):
    pdp_fare = pdp.pdp_isolate(model=model, dataset=dataset, model_features=dataset.columns, 
                               feature=feature, predict_kwds={})
    fig, axes = pdp.pdp_plot(pdp_fare, feature, x_quantile=True, show_percentile=True, 
                             plot_lines=True, plot_pts_dist=True)
    
  def interact_plot(self, model: Any, dataset: pd.DataFrame, features: List[str], plot_type: str="contour"):
    interact_data = pdp.pdp_interact(model=model, dataset=dataset, model_features=dataset.columns, 
                                     features=features, predict_kwds={})
    fig, axes = pdp.pdp_interact_plot(pdp_interact_out=interact_data, feature_names=features, 
                                      plot_type=plot_type, x_quantile=True, plot_pdp=True)
    
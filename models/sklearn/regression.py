from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, ARDRegression, RidgeCV
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression

class RegressionModel:
   def __init__(self, random_state=0):
      self.models = []
      self.models.append(("LinearRegression", 
                     Pipeline([
                               ("LinearRegression", LinearRegreesion())])))

    self.models.append(("Ridge",
                     Pipeline([
                               ("Ridge", Ridge(random_state=random_state))]))) 
    self.models.append(("Lasso", 
                    Pipeline([
                              ("Lasso", Lasso(random_state=random_state))]))) 
    self.models.append(("ElasticNet", 
                    Pipeline([
                              ("ElasticNet", ElasticNet(random_state=random_state))]))) 
    self.models.append(("SGDRegressor", 
                    Pipeline([
                              ("SGDRegressor", SGDRegressor(random_state=random_state))]))) 
    self.models.append(("PassiveAggressiveRegressor", 
                    Pipeline([
                              ("PassiveAggressiveRegressor", PassiveAggressiveRegressor(max_iter=100, tol=1e-3, random_state=random_state))]))) 

    self.models.append(("ExtraTreesRegressor", 
                     Pipeline([
                               ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=100, random_state=random_state))]))) 

    self.models.append(("AdaBoostRegressor", 
                     Pipeline([
                               ("AdaBoostRegressor", AdaBoostRegressor(n_estimators=200,
                                                                       random_state=random_state))]))) 

    self.models.append(("GradientBoostingRegressor", 
                     Pipeline([
                               ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=random_state))]))) 
    self.models.append(("KNeighborsRegressor", 
                     Pipeline([
                               ("KNeighborsRegressor", KNeighborsRegressor(n_neighbors=3))])))

    self.models.append(("PLSRegression",
                     Pipeline([
                               ("PLSRegression", PLSRegression(n_components=10))])))

    self.models.append(("SVR",
                     Pipeline([
                               ("SVR", SVR(kernel="rbf", C=1e3, gamma=0.1, epsilon=0.1))])))
      
      

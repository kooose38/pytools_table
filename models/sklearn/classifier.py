from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

class ClassifierModel:
  def __init__(self, random_state=0):
    self.models = []
    self.models.append(("LogReg", 
                     Pipeline([
                               ("LogReg", LogisticRegression(n_jobs=-1, random_state=random_state))])))

    self.models.append(("XGBClassifier",
                     Pipeline([
                               ("XGB", XGBClassifier(n_jobs=-1, random_state=random_state))]))) 
    self.models.append(("SVC", 
                    Pipeline([
                              ("KNN", SVC(random_state=random_state))]))) 
    self.models.append(("LogisticRegression", 
                    Pipeline([
                              ("LogisticRegression", LogisticRegression(random_state=random_state))]))) 
    self.models.append(("SGD", 
                    Pipeline([
                              ("SGD", SGDClassifier(random_state=random_state))]))) 
    self.models.append(("LinearSVC", 
                    Pipeline([
                              ("LinearSVC", LinearSVC(random_state=random_state))]))) 

    self.models.append(("DecisionTreeClassifier", 
                     Pipeline([
                               ("DecisionTrees", DecisionTreeClassifier(random_state=random_state))]))) 

    self.models.append(("RandomForestClassifier", 
                     Pipeline([
                               ("RandomForest", RandomForestClassifier(n_estimators=200, n_jobs=-1, 
                                                                       random_state=random_state))]))) 

    self.models.append(("GradientBoostingClassifier", 
                     Pipeline([
                               ("GradientBoosting", GradientBoostingClassifier(n_estimators=200,
                                                                               random_state=random_state))]))) 
    self.models.append(("RidgeClassifier", 
                     Pipeline([
                               ("RidgeClassifier", RidgeClassifier(random_state=random_state))])))

    self.models.append(("BaggingRidgeClassifier",
                     Pipeline([
                               ("BaggingClassifier", BaggingClassifier(n_jobs=-1, random_state=random_state))])))

    self.models.append(("ExtraTreesClassifier",
                     Pipeline([
                               ("ExtraTrees", ExtraTreesClassifier(n_jobs=-1, random_state=random_state))])))
    
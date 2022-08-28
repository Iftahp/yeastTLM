"""
Models used
"""

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from constants import RANDOM_STATE_LRCV


# models to compare and select from, no random state  - to allow some variability in model selection
MODELS = [
            ('XGB', XGBClassifier(n_estimators=500,
                                  max_depth=3,
                                  max_features='log2',
                                  eval_metric='logloss',
                                  learning_rate=0.1)),
            ('RF', RandomForestClassifier(n_estimators=500,
                                          criterion='entropy',
                                          max_features=0.25)),
            ('ET', ExtraTreesClassifier(n_estimators=1000,
                                        max_features='log2',
                                        criterion='entropy')),
            ('PSVC', SVC(C=0.01,
                         gamma=0.1,
                         degree=3,
                         coef0=10,
                         kernel='poly')),
            ('LSVC', LinearSVC(max_iter=100000)),
            ('LRCV', LogisticRegressionCV(max_iter=10000))
 ]

# final model that was selected (random state for reproducibility)
LRCV = LogisticRegressionCV(random_state=RANDOM_STATE_LRCV, max_iter=10000)
LRCV_ORTHOLOGS = LogisticRegressionCV(random_state=RANDOM_STATE_LRCV, class_weight='balanced', max_iter=10000)

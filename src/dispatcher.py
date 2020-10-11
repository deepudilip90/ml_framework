from sklearn import ensemble, tree
MODELS={
    'randomforest': ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    'extratrees': ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    'decision_tree_gini': tree.DecisionTreeClassifier(criterion='gini'),
    'decision_tree_entropy': tree.DecisionTreeClassifier(criterion='entropy')
    # 'randomforest': ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2)

}
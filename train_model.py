from idpanel.training.vectorization import load_raw_feature_vectors
from idpanel.training.features import load_raw_features
from idpanel.labels import load_labels
from idpanel.training.prevectorization import load_panel_paths
from idpanel.classification import ClassificationEngine
from idpanel.blacklist import feature_blacklist
from idpanel.decision_tree import DecisionTree
from sklearn.cross_validation import cross_val_score, train_test_split
import json
import numpy as np

decision_trees = {}
if __name__ == "__main__" or True:
    # todo Add some command line options
    # Output path for model
    # Maximum number of attempts to generate a model for each label
    maximum_model_attempts = 100
    # Maximum number of models per label
    max_models_per_label = 3

    label_indeces = load_labels()
    raw_features = load_raw_features()
    original_labels, names, vectors = load_raw_feature_vectors()
    labels = [label_indeces.index(l) for l in original_labels]

    vectors = np.array(vectors)
    print "Creating training and testing sets"
    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, stratify=labels)
    print X_train.shape[0], "samples in training set,", len(set(list(y_train))), "labels in training set"
    print X_test.shape[0], "samples in training set,", len(set(list(y_test))), "labels in testing set"

    decision_trees = {}
    for label in label_indeces:
        if label == "not_panel":
            continue
        temp_paths = load_panel_paths(label)
        paths = set()
        for path in temp_paths:
            skip = False
            for bl in feature_blacklist:
                if bl in path:
                    skip = True
                    break
            if not skip:
                paths.add(path)

        tX_train = X_train.copy()
        tX_test = X_test.copy()
        tvectors = vectors.copy()
        allowed_features = set()
        for rfi, rf in enumerate(raw_features):
            if rf[0] not in paths or rf[1] in [403, 401, 302, 404]:
                tvectors[:, rfi] = 0
                tX_test[:, rfi] = 0
                tX_train[:, rfi] = 0
            else:
                variance = tvectors[:, rfi].var()
                if variance != 0:
                    allowed_features.add(rfi)

        print ""
        print ""
        print ""
        print "Building decision tree for {0}".format(label)
        print "{0} acceptable features after filters".format(len(allowed_features))
        label_index = label_indeces.index(label)
        ty_train = [1 if label_index == i else 0 for i in y_train]
        ty_test = [1 if label_index == i else 0 for i in y_test]
        tlabels = [1 if label_index == i else 0 for i in labels]

        best_model = None
        best_result = 0

        model_results = []

        for trial in xrange(maximum_model_attempts):
            if len(allowed_features) == 0:
                print "Exhausted available features"
                break
            dt = DecisionTree(allowed_features)
            dt.fit(tX_train, ty_train)
            #print dt.tree

            test_results = dt.score(vectors, tlabels)
            model_results.append((dt, test_results))

            allowed_features -= set(dt.features_used)

            if len(model_results) > max_models_per_label:
                model_results = sorted(model_results, key=lambda x: x[1], reverse=True)[:max_models_per_label]

            if len(model_results) >= 3:
                can_stop = True
                for mr in model_results:
                    if mr[1] != 1:
                        can_stop = False

                if can_stop:
                    break

        decision_trees[label] = []

        print "Best model for {0} results:".format(label)
        for mr in model_results:
            clf = mr[0]

            print "Score on training data:", clf.score(X_train, ty_train)
            print "Score on testing data:", clf.score(X_test, ty_test)
            print "Score on all data:", clf.score(vectors, tlabels)
            print ""

            predictions = clf.predict(vectors)
            for index in xrange(vectors.shape[0]):
                if predictions[index] != tlabels[index]:
                    print names[index], "detected as", predictions[index], "is actually", tlabels[index]

            print ""
            relevant_features = [(i, 0, raw_features[i]) for i in
                                 clf.features_used]
            print len(relevant_features), "features used in this decision tree"
            for rf in relevant_features:
                print rf

            print ""

            decision_trees[label].append(
                {"model": clf, "features": relevant_features}
            )

    sparse_features = []
    features_added = set()
    for label in decision_trees.keys():
        for model in decision_trees[label]:
            for feature in model["features"]:
                if feature[0] not in features_added:
                    features_added.add(feature[0])
                    sparse_features.append((feature[0], feature[2]))

    ce = ClassificationEngine(decision_trees, sparse_features, len(raw_features))
    print ce.get_required_requests()
    ce.save_model("bot_model.mdl")
    #for index in xrange(10):
    #    print ce.get_label_scores(None, vector=vectors[index, :])[0], original_labels[index]
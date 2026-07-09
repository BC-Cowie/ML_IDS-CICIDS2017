"""
src/train.py
Orchestrates training of all models in both binary and multiclass mode.
Call run_binary() or run_multiclass() directly, or use main.py.
"""

import os
import sys
import numpy as np
from sklearn.metrics import precision_recall_curve

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from src import preprocess, evaluate, visualise
from src.models import random_forest, xgboost_model, svm, ensemble


def _get_roc(model, X_test, y_test, binary=True):
    from sklearn.metrics import roc_curve
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if binary:
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, proba[:, 1])
            return fpr, tpr, auc
    return None


def run_binary(tune: bool = False):
    print("\n" + "="*60)
    print("  BINARY CLASSIFICATION — BENIGN vs ATTACK")
    print("="*60 + "\n")

    # ── Preprocess ──
    X_train, X_test, y_train, y_test, scaler, feature_names = \
        preprocess.run(mode="binary")

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Attack rate (test): {y_test.mean():.2%}\n")

    all_results = {}
    roc_data    = {}

    # ── Random Forest ──
    if config.RUN_RANDOM_FOREST:
        rf_model = random_forest.train(X_train, y_train, tune=tune)
        random_forest.save(rf_model)
        y_pred = rf_model.predict(X_test)
        y_prob = rf_model.predict_proba(X_test)
        metrics = evaluate.compute_metrics(y_test, y_pred, y_prob, binary=True)
        all_results["Random Forest"] = metrics
        evaluate.print_report(y_test, y_pred, ["BENIGN", "ATTACK"], "Random Forest")

        roc = _get_roc(rf_model, X_test, y_test)
        if roc: roc_data["Random Forest"] = roc

        # Feature importance
        fi = random_forest.feature_importances(rf_model, feature_names)
        visualise.plot_feature_importance(fi, model_name="Random Forest")
        visualise.plot_confusion_matrix(
            evaluate.get_confusion_matrix(y_test, y_pred),
            ["BENIGN", "ATTACK"], "Random Forest",
            filename="rf_confusion_matrix.png"
        )

        # PR tradeoff — useful for setting threshold
        prec, rec, thresh = precision_recall_curve(y_test, y_prob[:, 1])
        visualise.plot_precision_recall_tradeoff(
            list(prec), list(rec), list(thresh),
            model_name="Random Forest",
            filename="rf_pr_tradeoff.png"
        )

    # ── XGBoost ──
    if config.RUN_XGBOOST:
        xgb_model = xgboost_model.train(X_train, y_train, binary=True, tune=tune)
        xgboost_model.save(xgb_model)
        y_pred = xgb_model.predict(X_test)
        y_prob = xgb_model.predict_proba(X_test)
        metrics = evaluate.compute_metrics(y_test, y_pred, y_prob, binary=True)
        all_results["XGBoost"] = metrics
        evaluate.print_report(y_test, y_pred, ["BENIGN", "ATTACK"], "XGBoost")

        roc = _get_roc(xgb_model, X_test, y_test)
        if roc: roc_data["XGBoost"] = roc

        visualise.plot_confusion_matrix(
            evaluate.get_confusion_matrix(y_test, y_pred),
            ["BENIGN", "ATTACK"], "XGBoost",
            filename="xgb_confusion_matrix.png"
        )

    # ── SVM ──
    if config.RUN_SVM:
        svm_model = svm.train(X_train, y_train, tune=tune)
        svm.save(svm_model)
        y_pred = svm_model.predict(X_test)
        y_prob = svm_model.predict_proba(X_test)
        metrics = evaluate.compute_metrics(y_test, y_pred, y_prob, binary=True)
        all_results["SVM"] = metrics
        evaluate.print_report(y_test, y_pred, ["BENIGN", "ATTACK"], "SVM")

        roc = _get_roc(svm_model, X_test, y_test)
        if roc: roc_data["SVM"] = roc

        visualise.plot_confusion_matrix(
            evaluate.get_confusion_matrix(y_test, y_pred),
            ["BENIGN", "ATTACK"], "SVM",
            filename="svm_confusion_matrix.png"
        )

    # ── Neural Network ──
    if config.RUN_NEURAL_NET:
        from src.models.neural_network import NeuralNetworkModel
        # Split a val set from training for early stopping
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1,
            stratify=y_train, random_state=config.RANDOM_STATE
        )
        nn = NeuralNetworkModel(input_dim=X_train.shape[1], binary=True)
        nn.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        nn.save()
        y_pred = nn.predict(X_test)
        y_prob = nn.predict_proba(X_test)
        metrics = evaluate.compute_metrics(y_test, y_pred, y_prob, binary=True)
        all_results["Neural Network"] = metrics
        evaluate.print_report(y_test, y_pred, ["BENIGN", "ATTACK"], "Neural Network")

        roc = _get_roc(nn, X_test, y_test)
        if roc: roc_data["Neural Network"] = roc

        visualise.plot_confusion_matrix(
            evaluate.get_confusion_matrix(y_test, y_pred),
            ["BENIGN", "ATTACK"], "Neural Network",
            filename="nn_confusion_matrix.png"
        )

    # ── Ensemble ──
    if config.RUN_ENSEMBLE:
        # Build estimator list from saved sklearn models
        base_estimators = []
        try:
            base_estimators.append(("rf",  random_forest.load()))
        except Exception: pass
        try:
            base_estimators.append(("xgb", xgboost_model.load()))
        except Exception: pass
        try:
            base_estimators.append(("svm", svm.load()))
        except Exception: pass

        if len(base_estimators) >= 2:
            stack = ensemble.train_stacking(X_train, y_train, base_estimators)
            ensemble.save(stack, "stacking")
            y_pred = stack.predict(X_test)
            y_prob = stack.predict_proba(X_test)
            metrics = evaluate.compute_metrics(y_test, y_pred, y_prob, binary=True)
            all_results["Stacking Ensemble"] = metrics
            evaluate.print_report(y_test, y_pred, ["BENIGN", "ATTACK"], "Stacking Ensemble")

            visualise.plot_confusion_matrix(
                evaluate.get_confusion_matrix(y_test, y_pred),
                ["BENIGN", "ATTACK"], "Stacking Ensemble",
                filename="ensemble_confusion_matrix.png"
            )

    # ── Summary ──
    evaluate.summarise_all(all_results, binary=True)
    evaluate.save_results(all_results, "binary_results.json")
    visualise.plot_model_comparison(all_results, filename="binary_model_comparison.png")
    if roc_data:
        visualise.plot_roc_curves(roc_data, filename="binary_roc_curves.png")

    return all_results


def run_multiclass(tune: bool = False):
    print("\n" + "="*60)
    print("  MULTICLASS CLASSIFICATION — Attack Type Identification")
    print("="*60 + "\n")

    result = preprocess.run(mode="multiclass")
    X_train, X_test, y_train, y_test, scaler, feature_names, le = result
    class_names = list(le.classes_)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Classes: {class_names}\n")

    all_results = {}

    # ── Random Forest ──
    if config.RUN_RANDOM_FOREST:
        rf_model = random_forest.train(X_train, y_train, tune=tune)
        y_pred = rf_model.predict(X_test)
        y_prob = rf_model.predict_proba(X_test)
        metrics = evaluate.compute_metrics(y_test, y_pred, y_prob, binary=False)
        all_results["Random Forest"] = metrics
        evaluate.print_report(y_test, y_pred, class_names, "Random Forest (Multiclass)")
        visualise.plot_confusion_matrix(
            evaluate.get_confusion_matrix(y_test, y_pred),
            class_names, "Random Forest",
            filename="mc_rf_confusion_matrix.png"
        )

    # ── XGBoost ──
    if config.RUN_XGBOOST:
        xgb_m = xgboost_model.train(X_train, y_train, binary=False, tune=tune)
        y_pred = xgb_m.predict(X_test)
        y_prob = xgb_m.predict_proba(X_test)
        metrics = evaluate.compute_metrics(y_test, y_pred, y_prob, binary=False)
        all_results["XGBoost"] = metrics
        evaluate.print_report(y_test, y_pred, class_names, "XGBoost (Multiclass)")

    # ── Neural Network ──
    if config.RUN_NEURAL_NET:
        from src.models.neural_network import NeuralNetworkModel
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1,
            stratify=y_train, random_state=config.RANDOM_STATE
        )
        nn = NeuralNetworkModel(input_dim=X_train.shape[1],
                                output_dim=len(class_names),
                                binary=False)
        nn.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        y_pred = nn.predict(X_test)
        y_prob = nn.predict_proba(X_test)
        metrics = evaluate.compute_metrics(y_test, y_pred, y_prob, binary=False)
        all_results["Neural Network"] = metrics
        evaluate.print_report(y_test, y_pred, class_names, "Neural Network (Multiclass)")

    evaluate.summarise_all(all_results, binary=False)
    evaluate.save_results(all_results, "multiclass_results.json")
    visualise.plot_model_comparison(all_results, filename="mc_model_comparison.png")

    return all_results, le

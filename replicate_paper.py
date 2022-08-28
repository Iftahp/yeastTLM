"""
Replicating the results of the paper
"""

from typing import Any
import features as fs
import helpers as hp
import constants as cs
import pandas as pd
from prettytable import PrettyTable
from models import MODELS, LRCV, LRCV_ORTHOLOGS
import os
from sklearn.metrics import auc, make_scorer, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import mannwhitneyu
import numpy as np
import pickle

# global def.
sns.set_theme(style="white")


def plot_mean_roc_auc(model_stats,
                      save_fig_path: str) -> None:
    """
    Plot AUC
    """

    plt.figure(figsize=(13, 8))
    ax = plt.axes()
    roc_auc = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for stat in model_stats:
        fpr, tpr, _ = stat['roc_curve']
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_auc.append(auc(fpr, tpr))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(roc_auc)
    ax.plot(mean_fpr, mean_tpr,
            label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
            lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05]
    )
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300)
    plt.show()


def get_spo_tlm_length_results(verbose: bool = True) -> tuple[list[dict[str, Any]], Any, Any, list[str]]:
    """
    Get the results for S. pombe 'short' vs. 'long' prediction and feature importance
    """

    scores_path = cs.SPO_TLM_SCORES
    features_nonzero_path = cs.FEATURE_IMPORTANCE_FEATURES_NONZERO
    try:
        with open(scores_path, 'rb') as s_path, open(features_nonzero_path, 'rb') as f_path:
            scores = pickle.load(s_path)
            features_nonzero = pickle.load(f_path)
        feature_importance = pd.read_excel(cs.FEATURE_IMPORTANCE).squeeze("columns")
        coef_nonzero = pd.read_excel(cs.FEATURE_IMPORTANCE_COEF_NONZERO).squeeze("columns")
    except FileNotFoundError:
        scores, df_coefs, df_importances_mean, all_runs_feat_idx = \
            hp.run_feature_importance_evaluation(df_feature=fs.get_tlm_length_final_features('spo'),
                                                 organism='spo',
                                                 verbose=verbose)
        # Coefficient of the features
        coef_ = df_coefs.mean()[all_runs_feat_idx]
        threshold = pd.Series(abs(coef_)).nlargest(n=15).iloc[-1]
        features_nonzero = coef_[abs(coef_) >= threshold].index.to_list()
        coef_nonzero = coef_[abs(coef_) >= threshold]
        feature_importance = df_importances_mean.mean()[features_nonzero]
        # save it
        with open(scores_path, 'wb') as s_path, open(features_nonzero_path, 'wb') as f_path:
            pickle.dump(scores, s_path, pickle.HIGHEST_PROTOCOL)
            pickle.dump(features_nonzero, f_path, pickle.HIGHEST_PROTOCOL)
        feature_importance.to_excel(cs.FEATURE_IMPORTANCE, index=False)
        coef_nonzero.to_excel(cs.FEATURE_IMPORTANCE_COEF_NONZERO, index=False)

    if verbose:
        print(f'{cs.BOLD}Results for S. pombe short vs. long classification{cs.END}', "Median scores:",
              pd.DataFrame(scores).select_dtypes(include=['float64']).median(), "Avg. scores:",
              pd.DataFrame(scores).select_dtypes(include=['float64']).mean(),
              "Telomere length dist. of short (0) vs long (1)",
              fs.get_tlm_length_final_features('spo')['tel_len'].value_counts().sort_index(), sep='\n')
    return scores, feature_importance, coef_nonzero, features_nonzero


def get_multi_class_results(verbose: bool = True) -> pd.DataFrame:
    """
    Get the results for a broader telomere phenotype categories (multi-class)
    """

    if os.path.isfile(cs.MULTI_CLASS_RESULTS):
        df_confusion = pd.read_excel(cs.MULTI_CLASS_RESULTS)
    else:
        # produce confusion matrix
        scores, results = hp.run_multi_class_evaluation(df_feature=fs.get_tlm_length_final_features('sce'),
                                                        verbose=verbose)
        df_confusion = pd.DataFrame([], columns=["y_pred", "y_true", "class"])
        for e in results:
            df_confusion = df_confusion.append(pd.DataFrame(np.c_[results[e]['y_pred'],
                                                                  results[e]['y_test'],
                                                                  results[e]['y_prob'],
                                                                  [cs.LABELS[label] for label in
                                                                   results[e]['y_multi']]],
                                                            columns=["y_pred", "y_true", "y_prob", "class"]),
                                               ignore_index=True)
        # have y_prob column reflect the probability of the class label
        df_confusion['y_prob'] = df_confusion['y_prob'].astype(float)
        df_confusion.loc[df_confusion['class'].isin(cs.LABELS[3:]), 'y_prob'] = 1 - df_confusion.loc[
            df_confusion['class'].isin(cs.LABELS[3:]), 'y_prob']
        df_confusion.to_excel(cs.MULTI_CLASS_RESULTS, index=False)
    if verbose:
        print("Phenotype distribution of the samples used:", f'{cs.BOLD}Raw counts:{cs.END}',
              df_confusion["class"].value_counts(),
              f'{cs.BOLD}Total samples:{cs.END}', df_confusion["class"].value_counts().sum(),
              f'{cs.BOLD}Percentages:{cs.END}',
              df_confusion["class"].value_counts() / df_confusion["class"].value_counts().sum() * 100, sep='\n')
    return df_confusion


def plot_boxplot(df_results: pd.DataFrame,
                 metric_name: str,
                 save_fig_path: str) -> None:
    """
    Plot and save boxplot of a given metric_name, based on some ordering
    """

    sort_labels = df_results.drop(columns='Model').median().sort_values(ascending=False).index.to_list()
    sns.boxplot(data=df_results, palette="flare", order=sort_labels)
    plt.xlabel('Features', fontweight='bold')
    plt.ylabel(metric_name, fontweight='bold')

    # statistical annotation
    stats = []
    x = df_results[sort_labels[0]].to_list()
    line_start = 1
    for j in range(1, len(sort_labels)):
        y = df_results[sort_labels[j]].to_list()
        _, pval = mannwhitneyu(x, y, alternative="greater")
        print(f"{sort_labels[0]} > {sort_labels[j]} has P-value {pval:.15f}",
              f"with median {metric_name} of {np.median(x)} vs. {np.median(y)}")
        if pval < 0.0001:
            stats.append((j, line_start))
        line_start += 0.2
    if stats:
        for x2, y in stats:
            x1, h = 0, 0.1
            plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
            plt.text((x1 + x2) * .5, y + h, "****", ha='center', va='bottom', color='k')

    sns.despine(offset=10, top=True, right=True)
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300)
    plt.show()


def plot_heatmap(order_by: str,
                 df_results: pd.DataFrame,
                 metric_name: str,
                 save_fig_path: str) -> None:
    """
    Plot and save heatmaps of a given metric_name, based on some ordering
    """

    # The results are grouped by the median across all test runs per model on each feature
    if order_by == 'models':
        # order the heatmap by the models' median score across all features
        sort_idx = df_results.groupby('Model').median().median(axis=1).sort_values(ascending=False).index
        df_heatmap = df_results.groupby('Model').median().loc[sort_idx, :]
    else:
        # order the heatmap by the features' median score across all models
        sort_idx = df_results.groupby('Model').median().median().sort_values(ascending=False).index
        df_heatmap = df_results.groupby('Model').median()[sort_idx]
    sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap="flare")
    plt.title(metric_name, fontsize=20, fontweight='bold')
    plt.xlabel('Features', fontweight='bold')
    plt.ylabel('Models', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300)
    plt.show()


def get_auc_and_mcc_results(features_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the results of AUC and MCC for a given feature type.
    """

    if features_type == 'single':
        auc_results_path = cs.SINGLE_FEATURE_AUC
        mcc_results_path = cs.SINGLE_FEATURE_MCC
    elif features_type == 'pairwise':
        auc_results_path = cs.PAIRWISE_FEATURES_AUC
        mcc_results_path = cs.PAIRWISE_FEATURES_MCC
    else:
        raise ValueError(f'feature type: {features_type} is not supported')
    if os.path.isfile(auc_results_path) and os.path.isfile(mcc_results_path):
        df_auc_results = pd.read_excel(auc_results_path)
        df_mcc_results = pd.read_excel(mcc_results_path)
    else:
        # get the feature set - no null rows or columns and with target column of telomere length
        features = fs.get_sce_tlm_length_single_features() if features_type == 'single' else \
            fs.get_sce_tlm_length_pairwise_features()
        # build the results and save them
        df_auc_results = hp.get_models_scores(models=MODELS, feature_list=features, verbose=True,
                                              metric='roc_auc')
        df_mcc_results = hp.get_models_scores(models=MODELS, feature_list=features, verbose=True,
                                              metric=make_scorer(matthews_corrcoef))
        df_auc_results.to_excel(auc_results_path, index=False)
        df_mcc_results.to_excel(mcc_results_path, index=False)
    return df_auc_results, df_mcc_results


def plot_various_telomere_phenotypes() -> None:
    """
    Plot various telomere phenotypes
    """

    df_confusion = get_multi_class_results(verbose=True)
    # plot samples used
    df_hist = df_confusion.copy()
    df_hist['class'] = pd.Categorical(df_hist['class'], cs.LABELS)
    df_hist['Category'] = df_hist['class'].apply(lambda cls: 'Short' if cls in cs.LABELS[:3] else "Long")
    sns.histplot(data=df_hist, x="class", hue='Category', palette="flare", discrete=True, stat='percent')
    plt.xticks(rotation=90)
    plt.xlabel('Telomere Length', fontweight='bold')
    plt.ylabel('Percent', fontweight='bold')
    plt.tight_layout()
    plt.savefig(cs.FIGURE_2C_1, dpi=300)
    plt.show()
    # plot  estimated probability
    sns.boxplot(x="class", y="y_prob", data=df_confusion, palette="flare", order=cs.LABELS)
    plt.xticks(rotation=90)
    plt.xlabel('Telomere Length', fontweight='bold')
    plt.ylabel('Prediction Probability Estimates', fontweight='bold')
    plt.tight_layout()
    plt.savefig(cs.FIGURE_2C_2, dpi=300)
    plt.show()
    # statistical tests
    for class_label in [cs.LABELS[:3], cs.LABELS[-1:2:-1]]:
        for i in range(len(class_label)):
            for j in range(i + 1, len(class_label)):
                x = df_confusion[df_confusion['class'] == class_label[i]]['y_prob'].to_list()
                y = df_confusion[df_confusion['class'] == class_label[j]]['y_prob'].to_list()
                _, pval = mannwhitneyu(x, y, alternative="greater")
                print(f"{class_label[i]} > {class_label[j]} has P-value {pval:.15f}",
                      f"with median probability estimates of {np.median(x)} vs. {np.median(y)}")


#######################################################################################################################
#                                                 Tables and Figures                                                  #
#######################################################################################################################
def table1() -> None:
    """
    Replicate Table 1
    """

    features = fs.get_sce_tlm_length_single_features() + fs.get_sce_tlm_length_pairwise_features()
    # subtract 2 from feature count, due to 'gene' and 'tel_len' columns
    table_data = [[feature[0],
                   hp.short_long(feature[1]['tel_len']).value_counts().sort_index()[0],
                   hp.short_long(feature[1]['tel_len']).value_counts().sort_index()[1],
                   feature[1].shape[0],
                   feature[1].shape[1] - 2]
                  for feature in features]
    headers = ['Feature name', 'Short TLM', 'Long TLM', 'Total', 'No. features']
    df_final_results = pd.DataFrame(table_data, columns=headers)
    df_final_results.to_excel(cs.TABLE_1, index=False)
    # PrettyTable to showcase all the final results
    summary_table = PrettyTable(headers)
    summary_table.add_rows(table_data)
    print(f'{cs.BOLD}Table 1:{cs.END}\n', summary_table)


def table3() -> None:
    """
    Replicate Table 3
    """

    sce_kegg_cyc2008_features, spo_kegg_cyc2008_features = fs.get_tlm_vs_non_ortholog_features()
    spo_e_1_4 = spo_kegg_cyc2008_features['tel_len'].value_counts()
    sce_e_1_4 = sce_kegg_cyc2008_features['tel_len'].value_counts()
    sce_spo_kegg_cyc2008_features = fs.get_orthologs_experiment_features(4)
    anchor_features = fs.get_orthologs_experiment_features(5)
    text = ('S.pombe', 'S. cerevisiae', 'Ortholog', 'KEGG, CYC2008', 'KEGG, CYC2008, Anchor genes')

    headers_1 = ['Method', 'Species', 'TLM genes', 'Non TLM genes', 'Total']
    table_data = [
        ['1', text[0], spo_e_1_4.iloc[1], spo_e_1_4.iloc[0], spo_e_1_4.iloc[1] + spo_e_1_4.iloc[0]],
        ['2', '', '', '', ''],
        ['3', text[1], sce_e_1_4.iloc[1], sce_e_1_4.iloc[0], sce_e_1_4.iloc[1] + sce_e_1_4.iloc[0]],
        ['4', '', '', '', ''],
        ['5', text[0], anchor_features['tel_len'].value_counts().iloc[1],
         anchor_features['tel_len'].value_counts().iloc[0], anchor_features.shape[0]]
    ]
    df_final_results_1 = pd.DataFrame(table_data, columns=headers_1)

    headers_2 = ['Method', 'Feature sets', 'No. of features']
    table_data = [
        ['1', text[2], '1'],
        ['2', text[3], sce_kegg_cyc2008_features.shape[1] - 2],  # remove 'gene' and 'tel_len'
        ['3', text[3], sce_kegg_cyc2008_features.shape[1] - 2],  # remove 'gene' and 'tel_len'
        ['4', text[3], sce_spo_kegg_cyc2008_features.shape[1] - 2],  # remove 'gene' and 'tel_len'
        ['5', text[4], anchor_features.shape[1] - 2]  # remove 'gene' and 'tel_len'
    ]
    df_final_results_2 = pd.DataFrame(table_data, columns=headers_2)
    # PrettyTable to showcase all the final results
    summary_table = PrettyTable(headers_1)
    summary_table.add_rows(df_final_results_1.values)
    print(f'{cs.BOLD}Table 3:{cs.END}\n', summary_table)
    summary_table = PrettyTable(headers_2)
    summary_table.add_rows(df_final_results_2.values)
    print(summary_table)
    with pd.ExcelWriter(cs.TABLE_3) as writer:
        df_final_results_1.to_excel(writer, sheet_name='Table 3_1', index=False)
        df_final_results_2.to_excel(writer, sheet_name='Table 3_2', index=False)


def table4() -> None:
    """
    Replicate Table 4
    """
    results_path = cs.TLM_VS_NON_RESULTS
    try:
        with open(results_path, 'rb') as r_path:
            table_data = pickle.load(r_path)
    except FileNotFoundError:
        pombe_cerev_df = hp.get_spo_to_sce_orthologs()
        train, test = fs.get_tlm_vs_non_ortholog_features()
        # experiment/method 1
        # from train (S. cerevisiae) get the matching ortholog phenotypes
        naive = pd.merge(train[['gene', 'tel_len']], pombe_cerev_df, left_on='gene', right_on='cerev')[['pombe', 'tel_len']]
        naive = naive.rename(columns={'tel_len': 'y_hat'})
        # match the ortholog phenotypes to the actual ones in test (S. pombe)
        naive = pd.merge(test[['gene', 'tel_len']], naive, left_on='gene', right_on='pombe')
        assert naive.shape[0] == test.shape[0], 'Naive model should have the same sample size!'
        y_test, y_hat = naive['tel_len'].values, naive['y_hat'].values
        # the model's prob. of predicting 1 is 1, so y_hat is our class 1 prob.
        table_data = [hp.orthologs_experiment_evaluation(y_test, y_hat, y_hat)]

        # experiment/method 2
        # define the pipeline and train on S. cerevisiae
        X, y = train.drop(columns=['tel_len', 'gene']).values, train['tel_len'].values
        sample_size, feature_size = X.shape
        pipe = hp.get_pipeline(estimator=LRCV_ORTHOLOGS, sample_size=sample_size, feature_size=feature_size)
        pipe.fit(X, y)
        # predict on test (S. pombe)
        X_test, y_test = test.drop(columns=['tel_len', 'gene']).values, test['tel_len'].values
        y_hat = pipe.predict(X_test)
        y_hat_prob = pipe.predict_proba(X_test)
        table_data.extend([hp.orthologs_experiment_evaluation(y_test, y_hat, y_hat_prob[:, 1])])

        # experiments/methods 3 - 5
        table_data.extend([hp.run_orthologs_cv_experiment(fs.get_orthologs_experiment_features(experiment), experiment)
                           for experiment in range(3, 6)])
        # save the results
        with open(results_path, 'wb') as r_path:
            pickle.dump(table_data, r_path, pickle.HIGHEST_PROTOCOL)
    # Save table and PrettyTable to showcase all the final results
    headers = ['Method', 'AUC', 'Recall', 'Precision']
    df_final_results = pd.DataFrame(table_data, columns=headers[1:])
    df_final_results[headers[0]] = np.arange(1, 6)
    df_final_results = df_final_results[headers]
    df_final_results.to_excel(cs.TABLE_4, index=False)
    summary_table = PrettyTable(headers)
    summary_table.add_rows(df_final_results.values)
    print(f'{cs.BOLD}Table 4:{cs.END}\n', summary_table)


def table_s1() -> None:
    """
    Replicate Table S1
    """

    names_map = {"gene": "Systematic Name", "name": "Gene Name", "tel_len": "Binary Phenotype"}
    # S. cerevisiae
    df_sce_tlm = hp.get_sce_tlm().rename(columns={"phenotype": "Original Phenotype"}).rename(columns=names_map)
    df_sce_tlm['Binary Phenotype'] = df_sce_tlm['Binary Phenotype'].apply(hp.short_long)
    # S. pombe
    df_spo_tlm = hp.get_spo_tlm().rename(columns=names_map)
    df_pombase_names = pd.read_table(cs.SPO_POMBASE_NAMES, header=0, names=['Gene', 'Name', 'Synonyms'])
    df_spo_tlm["Gene Name"] = df_spo_tlm["Systematic Name"].apply(hp.get_spo_gene_pombase, args=(df_pombase_names,))
    # both
    for df in (df_sce_tlm, df_spo_tlm):
        df['Binary Phenotype'] = df['Binary Phenotype'].apply(lambda b: 'short' if b == 0 else 'long')
        # PrettyTable to showcase all the final results
        summary_table = PrettyTable(df.columns.to_list())
        summary_table.add_rows(df.values)
        print(f'{cs.BOLD}Table S1:{cs.END}\n', summary_table)
    with pd.ExcelWriter(cs.TABLE_S1) as writer:
        df_sce_tlm.to_excel(writer, sheet_name='S. cerevisiae TLM genes', index=False)
        df_spo_tlm.to_excel(writer, sheet_name='S. pombe TLM genes', index=False)


def table_s2() -> None:
    """
    Replicate Table S2
    """

    df_final_results = hp.get_spo_to_sce_orthologs().rename(columns={"pombe": "S. pombe Systematic Name",
                                                                     "cerev": "S. cerevisiae Systematic Name"})
    df_final_results.to_excel(cs.TABLE_S2, index=False)
    # PrettyTable to showcase all the final results
    summary_table = PrettyTable(df_final_results.columns.to_list())
    summary_table.add_rows(df_final_results.values)
    print(f'{cs.BOLD}Table S2:{cs.END}\n', summary_table)


def table_s4() -> None:
    """
    Replicate Table S4
    """

    results_path = cs.SPO_LOO_RESULTS
    try:
        with open(results_path, 'rb') as r_path:
            results = pickle.load(r_path)
    except FileNotFoundError:
        results = hp.run_leave_one_out_predictions(fs.get_orthologs_experiment_features(5))
        # save it
        with open(results_path, 'wb') as r_path:
            pickle.dump(results, r_path, pickle.HIGHEST_PROTOCOL)

    # build final result table with all the needed data
    cerev_desc = pd.read_csv(cs.SCE_GENES_YEASTMINE, usecols=[0, 2, 3], names=['gene', 'gene_name', 'desc'])
    uniport_pombe = pd.read_excel(cs.SPO_UNIPROT, usecols=[1, 2])
    pombe_cerev_df = hp.get_spo_to_sce_orthologs()
    cerev_tlms = hp.get_sce_tlm()
    res = pd.DataFrame(results, columns=['gene', 'target', 'TLM probability'])
    res['gene'] = res['gene'].apply(lambda g: g[0])
    res['target'] = res['target'].apply(lambda t: t[0])
    res['TLM probability'] = res['TLM probability'].apply(lambda t: t[0][1])
    res['gene name'] = res['gene'].apply(hp.get_spo_gene_uniport, args=(uniport_pombe,))
    res.sort_values(by='TLM probability', ascending=False, inplace=True)
    df_tmp = pd.merge(res, pombe_cerev_df, left_on='gene', right_on='pombe').drop(columns='pombe')
    df_res = pd.merge(df_tmp, cerev_desc, left_on='cerev', right_on='gene',
                      suffixes=('_pombe', '_cerevisiae')).drop(columns='cerev')
    df_res['Is S. cerevisiae TLM'] = df_res['gene_cerevisiae'].isin(cerev_tlms['gene'])
    df_res.sort_values(by='TLM probability', ascending=False, inplace=True)
    # Renaming
    df_res.rename(columns={"gene name": "Gene Name S. pombe",
                           "gene_name": "Gene Name S. cerevisiae",
                           "gene_cerevisiae": "Systematic Name S. cerevisiae",
                           "gene_pombe": "Systematic Name S. pombe",
                           "TLM probability": "S. pombe TLM Probability",
                           "desc": "Desc S. cerevisiae"}, inplace=True)
    # top 30 features that are not known to be S. pombe TLM genes
    df_res = df_res[df_res['target'] == 0].head(30).drop(columns=['target'])
    # save it
    df_res.to_excel(cs.TABLE_S4, index=False)
    # PrettyTable to showcase all the final results
    summary_table = PrettyTable(df_res.columns.to_list())
    summary_table.add_rows(df_res.values)
    print(f'{cs.BOLD}Table S4:{cs.END}\n', summary_table)


def figure1() -> None:
    """
    Replicate Figure 1
    """

    df_auc_results, df_mcc_results = get_auc_and_mcc_results(features_type='single')
    # plot
    plot_heatmap(order_by='models', df_results=df_auc_results, metric_name='AUC', save_fig_path=cs.FIGURE_1A_AUC)
    plot_heatmap(order_by='models', df_results=df_mcc_results, metric_name='MCC', save_fig_path=cs.FIGURE_1A_MCC)
    plot_boxplot(df_results=df_auc_results, metric_name='AUC', save_fig_path=cs.FIGURE_1B_AUC)
    plot_boxplot(df_results=df_mcc_results, metric_name='MCC', save_fig_path=cs.FIGURE_1B_MCC)


def figure2() -> None:
    """
    Replicate Figure 2
    """

    df_auc_results, df_mcc_results = get_auc_and_mcc_results(features_type='pairwise')
    # plot AUC and MCC
    plot_heatmap(order_by='features', df_results=df_auc_results, metric_name='AUC', save_fig_path=cs.FIGURE_2A_AUC)
    plot_heatmap(order_by='features', df_results=df_mcc_results, metric_name='MCC', save_fig_path=cs.FIGURE_2A_MCC)
    # plot scatter
    df_mcc_med = df_mcc_results.groupby('Model').median()
    df_auc_med = df_auc_results.groupby('Model').median()
    assert np.all(df_auc_med.index == df_mcc_med.index)
    scatter_data = {"MCC": [], "AUC": [], "Features": [], "Model": []}
    for col in df_auc_med.columns:
        scatter_data["MCC"].extend(df_mcc_med[col].to_list())
        scatter_data["AUC"].extend(df_auc_med[col].to_list())
        scatter_data["Features"].extend([col for _ in range(df_auc_med[col].shape[0])])
        scatter_data["Model"].extend(df_auc_med[col].index.to_list())
    df_scatter = pd.DataFrame(scatter_data)
    g1 = sns.relplot(data=df_scatter, x="MCC", y="AUC", hue="Model", style="Features", legend="brief", palette="flare")
    g1.ax.add_patch(patches.Rectangle(xy=(0.49, 0.81), width=0.03, height=0.03, linewidth=1, color='k', fill=False))
    g1.despine(left=True, bottom=True)
    g1.ax.set_xlabel('MCC', fontweight='bold')
    g1.ax.set_ylabel('AUC', fontweight='bold')
    plt.savefig(cs.FIGURE_2B_1, dpi=300)
    plt.show()
    # plot scatter enlarged
    g2 = sns.relplot(data=df_scatter[(df_scatter['AUC'] > 0.8) & (df_scatter['MCC'] > 0.5)],
                     x="MCC", y="AUC", s=100, hue="Model", style="Features", palette="flare")
    g2.despine(left=True, bottom=True)
    g2.ax.set_xlabel('MCC', fontweight='bold')
    g2.ax.set_ylabel('AUC', fontweight='bold')
    plt.tight_layout()
    plt.savefig(cs.FIGURE_2B_2, dpi=300)
    plt.show()
    # plot various telomere phenotypes
    plot_various_telomere_phenotypes()


def figure3() -> None:
    """
    Replicate Figure 3
    """

    scores, feature_importance, coef_nonzero, features_nonzero = get_spo_tlm_length_results()
    plot_mean_roc_auc(model_stats=scores, save_fig_path=cs.FIGURE_3A)
    _, pathways_desc = hp.get_kegg_pathways(organism="spo")
    complex_map = hp.get_spo_complex_names()
    # plot 3B part 1
    plt.figure(figsize=(15, 4))
    plt.plot(coef_nonzero, 'o')
    xlabels = [complex_map[feat] if feat[0] != 'p'
               else pathways_desc['path:spo' + feat.split(':')[1]][0].replace('- Schizosaccharomyces pombe (fission '
                                                                              'yeast)', '') for feat in
               features_nonzero]
    plt.xticks(np.arange(len(coef_nonzero)), xlabels, rotation=90)
    plt.xlabel("Feature Name", fontweight='bold')
    plt.ylabel("Coefficient Magnitude", fontweight='bold')
    plt.savefig(cs.FIGURE_3B_1, dpi=300, bbox_inches='tight')
    plt.show()
    # plot 3B part 2
    plt.figure(figsize=(15, 2))
    plt.plot(feature_importance, 'o')
    plt.xticks(np.arange(len(feature_importance)),
               xlabels,
               rotation=90)
    # remove xticks
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are on
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.ylabel("Permutation Importance", fontweight='bold')
    plt.savefig(cs.FIGURE_3B_2, dpi=300)
    plt.show()


def main() -> None:
    """
    Replicate figures and tables
    """

    # make sure all the folders are in place:
    assert os.path.isdir('data') and os.path.isdir('features') and os.path.isdir('results') \
           and os.path.isdir('tables') and os.path.isdir('figures'), 'The following folders need to exist:' \
                                                                     ' data, features, results, tables and figures.' \
                                                                     ' All, but the data folder could be empty.'
    # Figures
    figure1()
    figure2()
    figure3()

    # Tables
    table1()
    table3()
    table4()

    # Supplementary Tables
    table_s1()
    table_s2()
    table_s4()


if __name__ == '__main__':
    main()


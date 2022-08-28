"""
Features used in the paper
"""

from typing import Any
import helpers as hp
import pickle5 as pickle
import os
import pandas as pd
import constants as cs


def get_kegg_features(organism: str):
    """
    For the given organisms: sce (Saccharomyces cerevisiae [S288c]) and
    spo (Schizosaccharomyces pombe [972h]) return the KEGG features based on GI scores.
    """

    data_path = cs.SCE_KEGG_FEATURES if organism == 'sce' else cs.SPO_KEGG_FEATURES
    if os.path.isfile(data_path):
        return pd.read_pickle(data_path)
    # get pathway to genes dict
    pathways_to_genes, _ = hp.get_kegg_pathways(organism)
    # get the GI scores
    df_gi = hp.get_gi_scores(organism)
    # produce the features
    df_kegg_features = hp.get_proportional_features(df_gi, pathways_to_genes)
    # have the column names consistent for both organisms
    df_kegg_features.rename(columns={col: col.replace(organism, "")
                                     for col in df_kegg_features.columns if col != 'gene'},
                            inplace=True)
    # pickle and return them
    df_kegg_features.to_pickle(data_path)
    return df_kegg_features


def get_go_features(namespace: str,
                    low: int = 3,
                    high: int = 30) -> pd.DataFrame:
    """
    GO [BP or CC] features.
    For GO BP, we considered for each gene its proportion from the genes of GO BP terms
    that have a non zero GI score with it.
    For GO CC the features indicate for each protein complex a gene’s membership in it.
    The GO terms contain genes of size between low and high. The gene considered are
    the ones that are in the GI data.
    Returns a DataFrame of features, having a 'gene' and GO term columns.
    """

    if namespace == 'BP':
        data_path = f'{cs.SCE_GO_BP_FEATURES}_{low}_{high}.pkl'
    elif namespace == 'CC':
        data_path = f'{cs.SCE_GO_CC_FEATURES}_{low}_{high}.pkl'
    else:
        raise ValueError(f'namespace: {namespace} is not supported')
    if os.path.isfile(data_path):
        return pd.read_pickle(data_path)
    # get a dict GO term to its genes, for terms with low to high number of genes
    go_to_genes = hp.get_go_to_genes(namespace=namespace, low=low, high=high)
    # get the GI scores
    df_gi = hp.get_gi_scores('sce')
    # produce the features
    if namespace == 'BP':
        df_go_features = hp.get_proportional_features(df_score=df_gi,
                                                      gene_groups=go_to_genes)
    else:  # 'CC'
        df_go_features = hp.get_membership_features(genes=set(df_gi['gene'].values),
                                                    gene_groups=go_to_genes)
    # pickle and return them
    df_go_features.to_pickle(data_path)
    return df_go_features


def get_curated_complexes_features(organism: str) -> pd.DataFrame:
    """
    For the given organisms: sce (Saccharomyces cerevisiae [S288c]) and
    spo (Schizosaccharomyces pombe [972h]) return the complexes from CYC2008 and PomBase, respectively.
    """

    if organism == 'sce':  # CYC2008
        data_path = cs.SCE_CYC2008_FEATURES
    elif organism == 'spo':  # PomBase
        data_path = cs.SPO_POMBASE_COMPLEX_FEATURES
    else:
        raise ValueError(f'Organism: {organism} is not supported')
    if os.path.isfile(data_path):
        return pd.read_pickle(data_path)
    # get complexes dict, the column names are consistent for both organisms, with GO IDs (non unique ones are marked)
    complex_to_genes = hp.get_sce_complexes() if organism == 'sce' else hp.get_spo_complexes()
    # genes for features
    gi_gene_set = set(hp.get_gi_scores(organism)['gene'].values)
    tlm_genes = hp.get_sce_tlm() if organism == 'sce' else hp.get_spo_tlm()
    gene_set = gi_gene_set | set(tlm_genes['gene'].values)
    # build features
    df_complex_features = hp.get_membership_features(genes=gene_set,
                                                     gene_groups=complex_to_genes)
    # pickle and return them
    df_complex_features.to_pickle(data_path)
    return df_complex_features


def get_tlm_vs_non_ortholog_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the features for both yeasts, used in the orthologs context.
    The features include a gene and target column and have no null columns and rows.
    """

    # build complex and KEGG features
    df_spo_features = pd.merge(get_curated_complexes_features('spo'), get_kegg_features('spo'))
    df_sce_features = pd.merge(get_curated_complexes_features('sce'), get_kegg_features('sce'))
    # remove null columns and rows
    for df in [df_spo_features, df_sce_features]:
        df = df.loc[:, (df != 0).any(axis=0)]  # Remove zero columns
        df = df.loc[~(df.drop(columns=['gene']) == 0).all(1)]  # Remove zero rows
    # leave only feature in both
    matching_features = set(df_spo_features.columns).intersection(df_sce_features.columns)
    df_spo_features = df_spo_features[list(matching_features)]
    df_sce_features = df_sce_features[list(matching_features)]
    # add target
    for df, df_tlm in zip([df_spo_features, df_sce_features], [hp.get_spo_tlm(), hp.get_sce_tlm()]):
        df['tel_len'] = 0
        df.loc[df['gene'].isin(df_tlm['gene']), ['tel_len']] = 1
    # get the orthologs
    pombe_cerev_df = hp.get_spo_to_sce_orthologs()
    # reduce the orthologs to genes we have features in both yeasts
    orthologs = pd.merge(df_spo_features['gene'], pombe_cerev_df, left_on='gene', right_on='pombe').drop(columns='gene')
    orthologs = pd.merge(df_sce_features['gene'], orthologs, left_on='gene', right_on='cerev').drop(columns='gene')
    # filter and return final features
    df_sce_features = df_sce_features[df_sce_features['gene'].isin(orthologs['cerev'].to_list())]
    df_spo_features = df_spo_features[df_spo_features['gene'].isin(orthologs['pombe'].to_list())]
    return df_sce_features, df_spo_features


def get_sce_tlm_length_single_features() -> list[list[Any]]:
    """
    Return A list of features for the task of binary telomere length classification in S. cerevisiae.
    They are:
    KEGG     - Proportional (percentage) KEGG based on GI
    GO BP    - GO BP based on propagation of GI scores
    CYC2008  - complex membership
    GO CC    - GO CC complex membership based on GI scores
    """

    # get S. cerevisiae tlm data
    df_sce_tlm = hp.get_sce_tlm()
    # get the features and add target column
    features = [
        ['GO BP', pd.merge(df_sce_tlm[['gene', 'tel_len']], get_go_features('BP'))],
        ['KEGG', pd.merge(df_sce_tlm[['gene', 'tel_len']], get_kegg_features('sce'))],
        ['CYC2008', pd.merge(df_sce_tlm[['gene', 'tel_len']], get_curated_complexes_features('sce'))],
        ['GO CC', pd.merge(df_sce_tlm[['gene', 'tel_len']], get_go_features('CC'))]
    ]
    # remove null columns and rows
    for feature in features:
        df = feature[1].copy()
        df = df.loc[:, (df != 0).any(axis=0)]  # Remove zero columns
        df = df.loc[~(df.drop(columns=['gene', 'tel_len']) == 0).all(1)]  # Remove zero rows
        feature[1] = df
    return features


def get_sce_tlm_length_pairwise_features() -> list[list[Any]]:
    """
    Return A list of feature pair sets for the task of binary telomere length classification in S. cerevisiae.
    """

    single_features = get_sce_tlm_length_single_features()
    n = len(single_features)
    features = list()
    for i in range(n):
        for j in range(i + 1, n):
            feature_name = f'{single_features[i][0]}/{single_features[j][0]}'
            df_feature = pd.merge(single_features[i][1], single_features[j][1], on=['gene', 'tel_len'])
            features.append([feature_name, df_feature])
    return features


def get_complex_and_kegg_names_in_both() -> list[str]:
    """
    Return matching features that we can map from S. cerevisiae complexes and pathways.
    Note, this is not necessarily the features used for the short vs. long task of S. cerevisiae, but
    the rather the data that is shared between the species.
    """

    # convert KEGG to the feature names used
    sce_pathways = {p.replace('sce', "") for p in hp.get_kegg_pathways('sce')[0].keys()}
    spo_pathways = {p.replace('spo', "") for p in hp.get_kegg_pathways('spo')[0].keys()}
    matching_pathways = list(sce_pathways.intersection(spo_pathways))
    matching_complexes = list(set(hp.get_sce_complexes().keys()).intersection(set(hp.get_spo_complexes().keys())))
    return matching_complexes + matching_pathways


def get_tlm_length_final_features(organism: str) -> pd.DataFrame:
    """
    For the given organisms: sce (Saccharomyces cerevisiae [S288c]) and spo (Schizosaccharomyces pombe [972h])
    return the final features used the task of binary telomere length classification (they are based on KEGG
    and CYC2008 complexes). The features include a target column and have no null columns and rows.
    """

    if organism == 'sce':
        sce_features = get_sce_tlm_length_pairwise_features()[3]
        assert sce_features[
                   0] == 'KEGG/CYC2008', 'the order of the features in the list matter. Especially, KEGG/CYC2008.' \
                                         ' See functions get_sce_tlm_length_pairwise_features() and ' \
                                         'get_sce_tlm_length_single_features() in features.py'
        return sce_features[1]
    elif organism == 'spo':
        # get S. pombe tlm data
        df_spo_tlm = hp.get_spo_tlm()
        # build the features
        df_spo_features = pd.merge(pd.merge(get_curated_complexes_features('spo'),
                                            df_spo_tlm[['gene', 'tel_len']], on=['gene']),
                                   get_kegg_features('spo'), on=['gene'])
        # only keep what is in both yeast datasets
        matching_feature_names = get_complex_and_kegg_names_in_both() + ['gene', 'tel_len']
        matching_features = list(set(matching_feature_names).intersection(set(df_spo_features.columns)))
        df_spo_features = df_spo_features.loc[:, matching_features]
        # Remove zero columns and rows
        df_spo_features = df_spo_features.loc[:, (df_spo_features != 0).any(axis=0)]
        df_spo_features = df_spo_features.loc[~(df_spo_features.drop(columns=['gene', 'tel_len']) == 0).all(1)]
        return df_spo_features
    else:
        raise ValueError(f'Organism: {organism} is not supported')


def get_anchor_genes_propagation_features(organism: str,
                                          normalize: bool,
                                          anchor_genes: list[str]) -> pd.DataFrame:
    """
    For the given organisms: sce (Saccharomyces cerevisiae [S288c]) and
    spo (Schizosaccharomyces pombe [972h]) return the anchor gene features.
    """

    adj_mat = hp.get_adjacency_matrix(organism)
    prop_data = hp.propagation_pipeline(adj_mat=adj_mat.drop(columns='gene'),
                                        anchor_genes=anchor_genes,
                                        normalize=normalize)
    df_prop = pd.DataFrame(prop_data, columns=['anchor_score']).reset_index()
    df_prop['gene'] = df_prop['index']
    return df_prop.drop(columns=['index'])


def get_orthologs_experiment_features(experiment: int) -> pd.DataFrame:
    """
    Return the features per experiment/method in orthologs
    """

    df_sce_feature, df_spo_feature = get_tlm_vs_non_ortholog_features()
    if experiment == 3:
        return df_spo_feature
    elif experiment == 4:
        pombe_cerev_df = hp.get_spo_to_sce_orthologs()
        # for each S.pombe sample get its ortholog
        data = pd.merge(df_spo_feature, pombe_cerev_df, left_on='gene', right_on='pombe')
        # for each S.pombe sample get its ortholog (S.cerev) features without phenotypes
        return pd.merge(data, df_sce_feature, left_on='cerev', right_on='gene'). \
            drop(columns=['pombe', 'cerev', 'gene_y', 'tel_len_y']). \
            rename(columns={'gene_x': 'gene', 'tel_len_x': 'tel_len'})
    elif experiment == 5:
        df_prop_spo = get_anchor_genes_propagation_features(organism='spo',
                                                            normalize=False,
                                                            anchor_genes=list(cs.SPO_ANCHOR_GENES))
        return pd.merge(df_prop_spo, df_spo_feature)


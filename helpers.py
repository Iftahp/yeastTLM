"""
Routines to help all
"""

from typing import Any
from goatools.obo_parser import GODag
from goatools.anno.gaf_reader import GafReader
from Bio import pairwise2, SeqIO, ExPASy
from Bio.Align import substitution_matrices
import pandas as pd
import networkx as nx
import pickle5 as pickle
from bioservices.kegg import KEGG
import constants as cs
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import RepeatedKFold, cross_val_score, permutation_test_score, RepeatedStratifiedKFold, \
    LeaveOneOut
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import make_scorer, roc_curve, confusion_matrix, roc_auc_score, accuracy_score, \
    matthews_corrcoef, precision_score, recall_score
from sklearn.inspection import permutation_importance
from scipy.stats import sem
from models import LRCV, LRCV_ORTHOLOGS
from tqdm import tqdm

# useful lambdas for binarization of results
short_long = np.frompyfunc(lambda z: 0 if z <= 2 else 1, 1, 1)  # short will get the value zero(0) and long one(1)


# functions
def align(seq_a: Any,
          seq_b: Any) -> float:
    """
    Smith-Waterman alignment with default parameters (based on BLAST web API)
    """

    return pairwise2.align.localds(seq_a,
                                   seq_b,
                                   substitution_matrices.load("BLOSUM62"),
                                   -11,  # gap open
                                   -1,  # gap extend
                                   score_only=True)  # only get the best score


def get_alignment_score(s_gene: str,
                        y_gene: str,
                        s_uniport: pd.DataFrame,
                        y_uniport: pd.DataFrame) -> float:
    """
    Return alignment score between S.pombe and S.cerevisiae genes
    """
    # get the swissport id
    pombe = s_uniport[s_uniport['OLN'] == s_gene]['swissport'].values
    cerev = y_uniport[y_uniport['OLN'] == y_gene]['swissport'].values
    if len(pombe) != 1 or len(cerev) != 1:
        return 0
    # get fasta
    try:
        with ExPASy.get_sprot_raw(cerev[0]) as handle:
            cerev_s = SeqIO.read(handle, "swiss")
        with ExPASy.get_sprot_raw(pombe[0]) as handle:
            pombe_s = SeqIO.read(handle, "swiss")
    except ValueError:
        print(f"ValueError, cerev: {cerev[0]}   pombe: {pombe[0]}")
        return 0
    except Exception:
        print(f"Something else went wrong, cerev: {cerev[0]}   pombe: {pombe[0]}")
        return 0

    return align(pombe_s.seq, cerev_s.seq)


def get_spo_gene_pombase(gene: str,
                         spo_pombase: pd.DataFrame) -> str:
    """
    Retrieve S. pombe gene's name from PomBase
    """

    try:
        name = spo_pombase[spo_pombase['Gene'] == gene]['Name'].values[0]
        return name if name is not np.nan else gene
    except IndexError:
        return gene


def get_spo_gene_uniport(gene: str,
                         spo_uniport: pd.DataFrame) -> str:
    """
    Retrieve S. pombe gene's name from uniprot
    """

    try:
        idx = spo_uniport['OLN'].str.contains(gene, regex=False)
        name = spo_uniport[idx]['gene'].values[0]
        return name if name is not np.nan else gene
    except IndexError:
        return gene


def produce_spo_to_sce_orthologs(verbose: bool = True) -> pd.DataFrame:
    """
    Produce a DataFrame of S.pombe with S.cerevisiae orthologs
    """

    # init.
    data = list()
    cerev_uniport = pd.read_excel(cs.SCE_UNIPROT, usecols=['OLN', 'swissport'])
    pombe_uniport = pd.read_excel(cs.SPO_UNIPROT, usecols=['OLN', 'swissport'])
    # process the orthologs
    with open(cs.POMBASE_ORTHOLOGS, 'r') as reader:
        reader.readline()  # skip first line (it's a header)
        for line in reader:
            # cleanup of data and splitting into S. pombe and S. cerevisiae
            genes = line.replace('\n', '').replace('(N)', '').replace('(C)', '') \
                .replace('(FUSION-C)', '').replace('(FUSION-N)', '').split('\t')
            genes[0] = genes[0].replace(' ', '')  # pombe and trim spaces
            cerevs = genes[1].split('|')
            cerevs = [c.replace(' ', '') for c in cerevs if c != 'NONE']
            if verbose:
                print(genes[0])
            if len(cerevs) < 1:  # no ortholog
                continue
            elif len(cerevs) == 1:  # exactly 1 ortholog
                data.append({'pombe': genes[0], 'cerev': cerevs[0]})
            else:  # more than 1 ortholog
                max_score, max_cerev = 0, ''
                for c in cerevs:
                    aln_score = get_alignment_score(genes[0], c, pombe_uniport, cerev_uniport)
                    if aln_score >= max_score:
                        max_cerev = c
                        max_score = aln_score
                data.append({'pombe': genes[0], 'cerev': max_cerev})

    return pd.DataFrame(data)


def get_spo_to_sce_orthologs() -> pd.DataFrame:
    """
    Return a DataFrame of S.pombe (column named 'pombe') with S.cerevisiae (column named 'cerev') orthologs
    """

    if os.path.isfile(cs.SPO_TO_SCE_ORTHOLOGS):
        with open(cs.SPO_TO_SCE_ORTHOLOGS, "rb") as fh:
            df_spo_sce_orthologs = pickle.load(fh)
    else:
        df_spo_sce_orthologs = produce_spo_to_sce_orthologs()
        df_spo_sce_orthologs.to_pickle(cs.SPO_TO_SCE_ORTHOLOGS)
    return df_spo_sce_orthologs


def get_proportional_features(df_score: pd.DataFrame,
                              gene_groups: dict[str, list[str]]) -> pd.DataFrame:
    """
    For a given scoring DataFrame (having the column named 'gene' and the other columns are gene names)
    and a group name (keys) with its genes (values), return a DataFrame with proportional features and 'gene' column.
    """

    # assert 'gene' column
    assert 'gene' in df_score.columns
    # build features
    total = list()
    for _, row in df_score.iterrows():
        # find for current gene all the genes with scores that are non zero
        non_zero_genes = row.drop(index='gene')[row.drop(index='gene') != 0].index
        # calculate their percentage for all input gene groups
        ratio = [non_zero_genes.isin(gene_groups[p]).astype(int).sum() / len(gene_groups[p]) for p in gene_groups]
        total.append(ratio)
    # produce the dataframe with 'gene' column
    df_proportional = pd.DataFrame(total, columns=[p for p in gene_groups])
    df_proportional['gene'] = df_score['gene']
    return df_proportional


def get_membership_features(genes: set[str],
                            gene_groups: dict[str, list[str]]) -> pd.DataFrame:
    """
    For a given gene set and a group name (keys) with its genes (values),
    return a DataFrame with membership (values 1 or 0) features and 'gene' column.
    """

    memberships = {gene: ([int(gene in gene_groups[name]) for name in gene_groups]) for gene in genes}
    # have the columns as name of gene groups (e.g. complexes) and add a 'gene' column
    df_membership = pd.DataFrame(memberships).T
    df_membership['gene'] = df_membership.index
    df_membership.rename(columns={i: str(name) for i, name in enumerate(gene_groups)}, inplace=True)
    return df_membership.reset_index(drop=True)  # reset index and remove index column


def get_go_id_to_db_id(ogaf: GafReader,
                       evidence: set[str] = None,
                       go2geneids_flag: bool = True,
                       namespace: str = 'BP') -> dict:
    """
    For a given ogaf return a mapping of go terms (as keys)
    to db_ids (as values).
    """

    # get GO ID with its associated db_id
    if evidence is None:
        go2db_id = ogaf.get_id2gos(namespace=namespace, go2geneids=go2geneids_flag)
    else:
        go2db_id = ogaf.get_id2gos(namespace=namespace,
                                   go2geneids=go2geneids_flag,
                                   ev_include=evidence)
    return go2db_id


def get_children_gene_count(go_term: str,
                            godag: GODag,
                            go2db_id: dict) -> tuple[int, set]:
    """
    For a given godag and a go_term (GO ID) and a mapping of go terms (as keys)
    to db_ids (as values) return the count of db_ids up to this term
    (including the term itself) and the db_ids themselves. These db_ids are assumed to be
    the genes of that term.
    Note that the go2db_id mapping should match what is represented in the godag.
    """

    if go_term not in godag:
        return 0, set()
    # get all children (if when creating GODag we do NOT
    # pass `optional_attrs` parameter then it is an 'is_a' graph)
    c = godag[go_term].get_all_children()
    # sum all the unique genes of children
    uniq_genes = set()
    for go_id in c:
        uniq_genes.update(go2db_id.get(go_id, []))
    # add the genes that are in the GO ID itself
    uniq_genes.update(go2db_id.get(go_term, []))
    return len(uniq_genes), uniq_genes


def get_spo_gi_scores():
    """
    Get the **GI data** for Schizosaccharomyces pombe (fission yeast), taken from:
    https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-4.4.207/BIOGRID-ORGANISM-4.4.207.tab3.zip
    """

    pombe_interactions = pd.read_table(cs.SPO_BIOGRID, low_memory=False)
    df_pombe_gi = pombe_interactions[pombe_interactions['Experimental System Type'] == 'genetic']
    # we only want interactions within the organism
    yeast = 'Schizosaccharomyces pombe (972h)'
    df_pombe_gi = df_pombe_gi[(df_pombe_gi['Organism Name Interactor A'] == yeast) &
                              (df_pombe_gi['Organism Name Interactor B'] == yeast)]
    df_pombe_gi['gi'] = 1
    # Create Graph
    G = nx.from_pandas_edgelist(
        df_pombe_gi,
        source='Systematic Name Interactor A',
        target='Systematic Name Interactor B',
        edge_attr='gi'
    )
    # Build GI adjacency matrix
    return pd.DataFrame(
        nx.adjacency_matrix(G, weight='gi').todense(),
        index=G.nodes,
        columns=G.nodes
    ).reset_index().rename(columns={'index': 'gene'})


def get_gi_scores(organism: str) -> pd.DataFrame:
    """
    For the given organisms: sce (Saccharomyces cerevisiae [S288c]) and
    spo (Schizosaccharomyces pombe [972h]) return a DataFrame with GI scores and a 'gene' column.
    """

    if organism == 'sce':
        with open(cs.SCE_FULL_GI_SCORES, "rb") as fh:
            df_gi_scores = pickle.load(fh)
    elif organism == 'spo':
        df_gi_scores = get_spo_gi_scores()
    else:
        raise ValueError(f'Organism: {organism} is not supported')
    return df_gi_scores


def get_kegg_pathways(organism: str = 'sce') -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    For the given organisms: sce (Saccharomyces cerevisiae [S288c]) and
    spo (Schizosaccharomyces pombe [972h]) return the KEGG pathways and its genes dict and
    pathways with their descriptions dict.
    """

    # get the path
    if organism == 'sce':
        pathway_genes_path = cs.SCE_KEGG_GENES
        pathway_names_path = cs.SCE_KEGG_NAMES
    elif organism == 'spo':
        pathway_genes_path = cs.SPO_KEGG_GENES
        pathway_names_path = cs.SPO_KEGG_NAMES
    else:
        raise ValueError(f'Organism: {organism} is not supported')
    # get the data
    try:
        with open(pathway_genes_path, 'rb') as f_genes, open(pathway_names_path, 'rb') as f_names:
            pathway_genes = pickle.load(f_genes)
            pathway_names = pickle.load(f_names)
    # no data, build it
    # NOTE: using this to rebuild the data will not get the same KEGG release as used in the paper
    except FileNotFoundError:
        kegg = KEGG()
        kegg.organism = organism
        pathway_genes, pathway_names = {}, {}
        for pathway in kegg.pathwayIds:
            data = kegg.get(pathway)
            dict_data = kegg.parse(data)
            # We only care about low level pathways - that are comprised of genes and not of other pathways
            if 'GENE' in dict_data:
                pathway_genes[pathway] = list(dict_data['GENE'].keys())
                pathway_names[pathway] = dict_data['NAME']
        with open(pathway_genes_path, 'wb') as f_genes, open(pathway_names_path, 'wb') as f_names:
            pickle.dump(pathway_genes, f_genes, pickle.HIGHEST_PROTOCOL)
            pickle.dump(pathway_names, f_names, pickle.HIGHEST_PROTOCOL)
    return pathway_genes, pathway_names


def get_go_to_genes(namespace: str = 'BP',
                    low: int = 3,
                    high: int = 30) -> dict[str, list[str]]:
    """
    Return a dictionary with a GO ID (e.g. GO:0000433) as a key and the genes
    (e.g. YDR516C) as its value, for a given namespace ['BP', 'CC'].

    #########################################################################
    The DB IDs are filtered to contain only these that are in the GI set.
    #########################################################################

    Each GO ID contains genes of size between low and high.
    The gene count is calculated by counting all the genes of all the
    GO term children with an 'is_a' relation in the spanning tree with the GO term of
    interest as its root.
    """

    # get GO stuff
    godag = GODag(cs.GO_DAG_BASIC)  # no 'optional_attrs' means is_a only
    ogaf = GafReader(cs.SCE_GAF)
    # get GO ID to DB ID mapping
    go_to_dbids_yeast = get_go_id_to_db_id(ogaf=ogaf, namespace=namespace)
    # GI gene set
    gene_set_gi = set(get_gi_scores('sce')['gene'].values)
    # get gene to DB ID mapping
    gene_to_dbid = get_gene_to_db_id(gene_set_gi, ogaf)
    # relevant DB IDs
    dbids_gi = set(gene_to_dbid.values())
    goid_del = list()
    for goid in go_to_dbids_yeast:
        # keep only DB IDs from GI gene set
        go_to_dbids_yeast[goid] = dbids_gi.intersection(go_to_dbids_yeast[goid])
        # store GO IDs that don't meet the criteria
        go_term_count = get_children_gene_count(go_term=goid, godag=godag, go2db_id=go_to_dbids_yeast)[0]
        if len(go_to_dbids_yeast[goid]) == 0 or go_term_count < low or go_term_count > high:
            goid_del.append(goid)
    # filter out GO IDs
    for goid in goid_del:
        del go_to_dbids_yeast[goid]
    # return the result with a gene mapping and not DB ID
    dbid_to_gene = {gene_to_dbid[gene]: gene for gene in gene_to_dbid}
    return {goid: [dbid_to_gene[dbid] for dbid in go_to_dbids_yeast[goid]] for goid in go_to_dbids_yeast}


def get_gene_to_db_id(gene_set: set[str],
                      ogaf: GafReader) -> dict[str, str]:
    """
    Return a dict with a gene (e.g. YLR233C) as a key and the DB ID (e.g. S000005765) as its value.
    """

    assert isinstance(gene_set, set)
    # build the dict
    gene_db_id = {}
    for gene in gene_set:
        for gaf_data in ogaf.associations:
            if gene in gaf_data.DB_Synonym or gene in gaf_data.DB_Symbol:
                gene_db_id[gene] = gaf_data.DB_ID
                break  # we found the gene, no point to keep on searching
    return gene_db_id


def get_spo_complex_names() -> dict[str, list[str]]:
    """
    Return PomBase complex names. Their names are the values and the GO IDs are the keys (our internal naming)
    """

    # S. pombe data
    df_pombe_complexes = pd.read_table(cs.SPO_POMBASE_COMPLEXES)
    grouped = df_pombe_complexes.groupby('acc')
    return {g_name: group['GO_name'].values[0] for g_name, group in grouped}


def get_spo_complexes() -> dict[str, list[str]]:
    """
    Return PomBase complexes. Their names are GO IDs
    """

    # S. pombe data
    df_pombe_complexes = pd.read_table(cs.SPO_POMBASE_COMPLEXES)
    grouped = df_pombe_complexes.groupby('acc')
    # the set constructor is to avoid duplicate genes in a complex
    return {g_name: list(set(group['systematic_id'].values)) for g_name, group in grouped}


def get_sce_complexes() -> dict[str, list[str]]:
    """
    Return CYC2008 complexes. Their names are GO IDs, where non unique GO IDs are marked by '_' followed by the complex
    name.
    """

    # S. cerevisiae data
    df_comp = pd.read_excel(cs.SCE_CYC2008_COMPLEXES, usecols=[0, 2, 5])
    grouped = df_comp.groupby('Complex')
    # get GO ids that have one-to-one connection to complexes in CYC2008
    single_goid = pd.Series(df_comp['GO_id'].dropna().values).value_counts()
    single_goid = single_goid[single_goid == 1]
    unique_goid = single_goid.index.to_list()
    # the set constructor is to avoid duplicate genes in a complex
    unique_goid_complexes = {group['GO_id'].dropna().values[0]: list(set(group['ORF'].values))
                             for _, group in grouped if group['GO_id'].dropna().values[0]
                             in unique_goid}
    non_unique_goid_complexes = {f"{group['GO_id'].dropna().values[0]}_{name}": list(set(group['ORF'].values))
                                 for name, group in grouped if group['GO_id'].dropna().values[0]
                                 not in unique_goid}
    return unique_goid_complexes | non_unique_goid_complexes


def get_sce_tlm() -> pd.DataFrame:
    """
    Return S. cerevisiae TLM data. It has to have among others, the columns 'gene' and 'tel_len'.
    """

    return pd.read_excel(cs.SCE_TLM_DATA)


def get_spo_tlm() -> pd.DataFrame:
    """
    Return S. pombe TLM data. It has to have among others, the columns 'gene' and 'tel_len'.
    """

    if os.path.isfile(cs.SPO_TLM_DATA):
        return pd.read_excel(cs.SPO_TLM_DATA)
    # FYPO data
    # https://www.ontobee.org/search?ontology=FYPO&keywords=telomere+length&submit=Search+terms
    fypo = pd.read_table(cs.SPO_FYPO_DATA, low_memory=False)
    tel_pheno = {'FYPO:0002019': 1, 'FYPO:0002239': 0, 'FYPO:0006511': 0, 'FYPO:0003106': 0, 'FYPO:0003107': 0}
    tlm_pombe = fypo[fypo['FYPO ID'].isin(tel_pheno.keys())].iloc[:, [1, 2, 8]]
    tlm_pombe['tel_len'] = fypo['FYPO ID'].map(tel_pheno)
    tlm_pombe['tel_len'] = tlm_pombe['tel_len'].astype('int32')
    tlm_pombe.sort_values(by=['Gene systematic ID', 'tel_len'], inplace=True)
    tlm_pombe = tlm_pombe.drop_duplicates(subset=['Gene systematic ID'])
    tlm_pombe = tlm_pombe.rename(columns={"Gene systematic ID": "gene", "Gene name": "name"}).drop(columns='FYPO ID')
    # Liu et al. 2010 data https://doi.org/10.1038/cr.2010.107
    df_pombe_tlm = pd.read_excel(cs.SPO_LIU_DATA)
    df_pombe_tlm['tel_len'] = df_pombe_tlm['tel_len'].map({'VS': 0, 'S': 0, 'SS': 0,
                                                           'SL': 1, 'L': 1, 'VL': 1})
    res = pd.concat([df_pombe_tlm, tlm_pombe])

    # unify and clean data
    def change_to_lower_c(name):
        if name[-1] == 'C':
            return name[:-1] + 'c'
        return name

    res['name'] = res['name'].apply(change_to_lower_c)
    res['gene'] = res['gene'].apply(change_to_lower_c)
    # some genes that were mapped without systematic name
    res['gene'] = res['gene'].replace({'clr7': 'SPCC970.07c',
                                       'moc4': 'SPBC1718.07c',
                                       'rps17-2': 'SPCC24B10.09',
                                       'rps17-1': 'SPBC839.05c',
                                       'rpl27-1': 'SPBC685.07c',
                                       'rpl30': 'SPAC9G1.03c',
                                       'rpl18a-2': 'SPAC3A12.10',
                                       'rpl9-1': 'SPAC4G9.16c',
                                       'rps15a-1': 'SPAC22A12.04c',
                                       'mak10': 'SPBC1861.03',
                                       'mug80': 'SPBC1D7.03',
                                       'hos1': 'SPAC4C5.02c',
                                       'fta5': 'SPAC1F8.06'})
    res = res.drop_duplicates(subset=['gene'])
    assert res.shape[0] == len(set(res['gene'].to_list()))
    # save and return
    res.to_excel(cs.SPO_TLM_DATA, index=False)
    return res.reset_index()[['name', 'gene', 'tel_len']]


def get_pipeline(estimator: Any,
                 sample_size: int,
                 feature_size: int) -> Pipeline:
    """
    Return the Pipeline object used.
    """

    # more samples than features, just add standardization
    if sample_size > feature_size:
        pipe = make_pipeline(StandardScaler(), estimator)

    # more features than samples, standardization and then SelectFromModel with Bernoulli Naive Bayes Classifier
    else:
        pipe = make_pipeline(StandardScaler(),
                             SelectFromModel(estimator=BernoulliNB(),
                                             threshold=-np.inf,
                                             importance_getter='feature_log_prob_',
                                             max_features=sample_size),
                             estimator)
    return pipe


# evaluate a model with a given number of repeats
def evaluate_model(X: np.ndarray,
                   y: np.ndarray,
                   model: Any,
                   stratified=True,
                   score_func='accuracy') -> np.ndarray:
    """
    Return the results of the cs.REPEATS-repeated [stratified] cs.FOLDS-fold cross-validation experiments
    """

    # prepare the cross-validation procedure
    if stratified:
        cv = RepeatedStratifiedKFold(n_splits=cs.FOLDS,
                                     n_repeats=cs.REPEATS,
                                     random_state=cs.RANDOM_STATE_K_FOLDS)
    else:
        cv = RepeatedKFold(n_splits=splits, n_repeats=cs.REPEATS, random_state=cs.RANDOM_STATE_K_FOLDS)
    try:
        # evaluate the model
        scores = cross_val_score(model, X, y, scoring=score_func, cv=cv, n_jobs=-1)
    except:
        scores = cross_val_score(model, X, y, scoring=score_func, cv=cv)
    return scores


def binary_evaluation(y_test_bin: np.ndarray,
                      y_pred_bin: np.ndarray,
                      y_prob_bin: np.ndarray) -> dict[str, Any]:
    """
      Return measures for binary classification evaluation.

      :param y_test_bin: Ground truth (correct) labels

      :param y_pred_bin: Predicted labels, as returned by a classifier

      :param y_prob_bin: Target scores, can either be probability estimates of
                         the positive class, confidence values, or non-threshold
                         measure of decisions(as returned by "decision_function"
                         on some classifiers)

      :return:  dict with key having the name of the measure
                and its values are the measure scores

    """
    tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred_bin, labels=[0, 1]).ravel()
    return {'sn': tp / (tp + fn),
            'sp': tn / (tn + fp),
            'auc': roc_auc_score(y_test_bin, y_prob_bin),
            'roc_curve': roc_curve(y_test_bin, y_prob_bin),
            'acc': accuracy_score(y_test_bin, y_pred_bin),
            'mcc': matthews_corrcoef(y_test_bin, y_pred_bin)}


def run_feature_importance_evaluation(df_feature: pd.DataFrame,
                                      organism: str,
                                      verbose: bool = True) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame,
                                                                     Any]:
    """
    For the given feature dataframe, run our pipeline and return the results and scores for length prediction and
    feature importance obtained via a cs.REPEATS repeated stratified cs.FOLDS-fold cross-validation.
    """

    # split feat. and target, to both binary and multi-class
    X = df_feature.drop(columns=['tel_len', 'gene']).values
    y = short_long(df_feature['tel_len'].values).astype(int) if organism == 'sce' else df_feature[
        'tel_len'].values.astype(int)
    # init. params.
    folds_, rep_ = cs.FOLDS, cs.REPEATS
    sample_size, feature_size = X.shape
    scores = []
    feature_names = np.array(df_feature.drop(columns=['tel_len', 'gene']).columns)
    selected_features = pd.DataFrame(np.zeros((rep_ * folds_, feature_names.size)), columns=feature_names)
    feature_importance_mean = selected_features.copy()
    coefs = selected_features.copy()
    # run a cs.REPEATS repeated stratified cs.FOLDS-fold cross-validation
    rkf = RepeatedStratifiedKFold(n_splits=folds_, n_repeats=rep_, random_state=cs.RANDOM_STATE_K_FOLDS)
    for epoch, idx in enumerate(rkf.split(X, y)):
        # data split
        train_index, test_index = idx[0], idx[1]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # define the pipeline
        pipe = get_pipeline(estimator=LRCV, sample_size=sample_size, feature_size=feature_size)
        # train
        lr = pipe.fit(X_train, y_train)
        # calc. measures and keep for later
        scores.append(binary_evaluation(y_test,
                                        lr.predict(X_test),
                                        lr.decision_function(X_test)))
        feat_imp = permutation_importance(lr,
                                          X_test,
                                          y_test,
                                          n_repeats=cs.REPEATS,
                                          random_state=cs.RANDOM_STATE_PERMUTATION)
        feature_importance_mean.loc[epoch, :] = feat_imp.importances_mean
        # more samples than features, no selection in the pipeline
        if sample_size > feature_size:
            selected_features.loc[epoch, :] = 1
            coefs.loc[epoch, :] = lr.named_steps['logisticregressioncv'].coef_.flatten()
        # more features than samples we have selection
        else:
            selected_features.loc[epoch,
                                  feature_names[lr.named_steps.selectfrommodel.get_support()]] = 1
            coefs.loc[epoch,
                      feature_names[lr.named_steps.selectfrommodel.get_support()]] = \
                lr.named_steps['logisticregressioncv'].coef_.flatten()
    if verbose:
        print(f"No. samples {sample_size}, with {feature_size} features")
    # keep the feature indices that partake in all of the runs    
    sum_feat = selected_features.sum()
    all_runs_feat_idx = sum_feat[sum_feat == selected_features.shape[0]].index
    return scores, coefs, feature_importance_mean, all_runs_feat_idx


def run_multi_class_evaluation(df_feature: pd.DataFrame,
                               verbose: bool = True) -> tuple[list[dict[str, Any]], dict]:
    """
    For the given feature dataframe, run our pipeline and return the results and scores for
    a cs.REPEATS repeated stratified cs.FOLDS-fold cross-validation.
    """

    # split feat. and target, to both binary and multi-class
    X = df_feature.drop(columns=['tel_len', 'gene']).values
    y = short_long(df_feature['tel_len'].values).astype(int)
    y_multi = df_feature['tel_len'].values
    # init. params.
    folds_, rep_ = cs.FOLDS, cs.REPEATS
    sample_size, feature_size = X.shape
    scores = []
    results = {i: {'test_idx': 0, 'y_pred': 0, 'y_multi': 0, 'y_test': 0, 'y_prob': 0}
               for i in range(folds_ * rep_)}
    # run a cs.REPEATS repeated stratified cs.FOLDS-fold cross-validation
    rkf = RepeatedStratifiedKFold(n_splits=folds_, n_repeats=rep_, random_state=cs.RANDOM_STATE_K_FOLDS)
    for epoch, idx in enumerate(rkf.split(X, y)):
        # data split
        train_index, test_index = idx[0], idx[1]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # define the pipeline
        pipe = get_pipeline(estimator=LRCV, sample_size=sample_size, feature_size=feature_size)
        # train
        lr = pipe.fit(X_train, y_train)
        # predict
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)
        # calc. measures and keep for later
        results[epoch]['test_idx'] = test_index
        results[epoch]['y_pred'] = y_pred
        results[epoch]['y_prob'] = y_prob[:, 0]
        results[epoch]['y_multi'] = y_multi[test_index]
        results[epoch]['y_test'] = y_test
        scores.append(binary_evaluation(y_test,
                                        y_pred,
                                        lr.decision_function(X_test)))
    if verbose:
        # report the median of scores, selection size and phenotype dist.
        print(f"Median scores for {sample_size} selected features:")
        print(pd.DataFrame(scores).select_dtypes(include=['float64']).median())
        print(f"No. samples {sample_size}, with {feature_size} features")
        print("Telomere length dist.")
        print(df_feature['tel_len'].value_counts().sort_index())
        print("Dist. of short (0) vs long (1)")
        print(pd.Series(short_long(df_feature['tel_len'])).value_counts().sort_index())
    return scores, results


# complete model eval.
def build_and_score_pipeline(df_features: pd.DataFrame,
                             classifier: Any,
                             classifier_name: str,
                             name_of_features: str = '',
                             target_col: str = 'tel_len',
                             non_feature_cols: tuple[str, str] = ('tel_len', 'gene'),
                             stratified: bool = True,
                             binarize: bool = True,
                             score_func: str = 'roc_auc',  # https://bit.ly/38dnP4d and https://bit.ly/3k9P9mQ
                             verbose: bool = True) -> np.ndarray:
    """
    For a feature set and a model, create the pipeline and return the results of the cs.REPEATS-repeated
    [stratified] cs.FOLDS-fold cross-validation experiments. Use verbose to print results.
    """

    # init. vars and transform the target columns
    drop_col = [*non_feature_cols]
    X = df_features.drop(columns=drop_col).values
    y = short_long(df_features[target_col].values).astype(int) if binarize else \
        df_features[target_col].values.astype(int)
    sample_size, feature_size = X.shape
    # define the pipeline
    pipe = get_pipeline(estimator=classifier, sample_size=sample_size, feature_size=feature_size)
    # evaluate model
    n_scores = evaluate_model(X, y, pipe, stratified=stratified, score_func=score_func)
    # report performance
    if verbose:
        print(f'>{classifier_name} with the scoring function {score_func} for the feature set {name_of_features}:',
              f'Mean: {np.mean(n_scores):.3f}  Std: ({np.std(n_scores):.3f})  SE: {sem(n_scores):.3f}',
              f'Med: {np.median(n_scores):.3f}')
    return n_scores


def run_leave_one_out_predictions(df_features: pd.DataFrame) -> list[Any]:
    """
    leave-one-out to predict S. pombe TLM genes
    """

    # init. vars and transform the target columns
    X, y = df_features.drop(columns=['tel_len', 'gene']).values, df_features['tel_len'].values.astype(int)
    loo = LeaveOneOut()
    genes = df_features['gene'].values
    results = []
    # define the pipeline
    model = get_pipeline(estimator=LRCV_ORTHOLOGS, sample_size=X.shape[0], feature_size=X.shape[1])
    # Leave-One-Out cross-validator
    for train_index, test_index in tqdm(loo.split(X), total=loo.get_n_splits(X),
                                        desc=f'{cs.BOLD}Leave-One-Out{cs.END}'):
        # split the data to train-test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # fit
        model.fit(X_train, y_train)
        # predict
        results.append([genes[test_index], y_test, model.predict_proba(X_test)])
    return results


def get_models_scores(models: list[Any],
                      feature_list: list[list[Any]],
                      verbose: bool = True,
                      binarize: bool = True,
                      metric: str = 'roc_auc') -> pd.DataFrame:
    """
    For a list of features and a list of models, return the results of the cs.REPEATS-repeated
    [stratified] cs.FOLDS-fold cross-validation experiments with for the given metric. Use verbose to print results.
    """

    results = {feat[0]: [] for feat in feature_list}
    results['Model'] = []
    for m in models:
        results['Model'].extend([m[0] for _ in range(cs.REPEATS * cs.FOLDS)])
        for feat in feature_list:
            results[feat[0]].extend(build_and_score_pipeline(df_features=feat[1],
                                                             name_of_features=feat[0],
                                                             classifier_name=m[0],
                                                             score_func=metric,
                                                             verbose=verbose,
                                                             binarize=binarize,
                                                             classifier=m[1]))
    return pd.DataFrame(results)


def run_orthologs_cv_experiment(df_features: pd.DataFrame,
                                experiment: int) -> pd.Series:
    """
    Evaluate our system on S.pombe TLM and non TLM genes using cs.REPEATS repeated stratified cs.FOLDS-fold
    cross-validation and report the median results
    """

    scores = {'AUC': [], 'Recall': [], 'Precision': []}
    for metric_name, metric in zip(scores.keys(), ['roc_auc', 'recall', 'precision']):
        scores[metric_name] = build_and_score_pipeline(df_features=df_features,
                                                       name_of_features=f'Method {experiment} features',
                                                       classifier_name='Balanced LRCV',
                                                       score_func=metric,
                                                       verbose=True,
                                                       binarize=False,  # Use as-is, no need to binarize
                                                       classifier=LRCV_ORTHOLOGS)
    return pd.DataFrame(scores).median()


def get_weight_matrix(A: np.ndarray) -> np.ndarray:
    """
    Return W which is A*D^-1, unless D is singular, then return D^-0.5*A*D^-0.5

    :param A: Adjacency matrix
    :return: W --> weight_matrix
    """
    D = np.zeros_like(A)
    D[np.diag_indices(D.shape[0])] = np.count_nonzero(A, axis=1)

    try:
        return A @ np.linalg.inv(D)
    except np.linalg.LinAlgError:  # Singular matrix
        degree = np.count_nonzero(A, axis=1).astype(float)
        if np.count_nonzero(degree) != len(degree):  # zero divide
            degree[degree == 0] = np.inf
        D[np.diag_indices(D.shape[0])] = 1 / np.sqrt(degree)  # D^-(1/2)
        return D @ A @ D


def propagate(A: np.ndarray,
              p0: np.ndarray,
              alpha: float = 0.8,
              max_iter: int = 100,
              tol: float = 1e-4,
              W=None) -> tuple[Any, int]:
    """
    Propagate a network

    :param A: Adjacency matrix
    :param p0: The starting vector of scores representing our prior knowledge or experimental measurements
    :param alpha: Smoothing parameter
    :param max_iter: Maximum number of iterations
    :param tol: Relative tolerance with regards to norm of the difference between two consecutive iterations
                to declare convergence
    :param W: Weight matrix
    :return: p_ --> the state after max_iter or convergence; converged --> did we converged or not
    """

    if W is None:
        W = get_weight_matrix(A)

    alpha_p0 = alpha * p0
    one_minus_alpha_W = (1 - alpha) * W
    p_ = p0.copy()
    p_prev = p0.copy()
    converged = 0
    for i in range(max_iter):
        p_ = alpha_p0 + one_minus_alpha_W @ p_
        if np.linalg.norm(p_ - p_prev) < tol:
            converged = i
            break
        p_prev = p_

    return p_, converged


def propagation_pipeline(adj_mat: pd.DataFrame,
                         anchor_genes: list[str],
                         alpha_propagate: float = 0.2,
                         alpha_normalize: float = 0,
                         normalize: bool = True,
                         tol: float = 1e-7,
                         max_iter: int = 10_000_000,
                         verbose: bool = False) -> pd.Series:
    """
    Produce propagation features to given anchor genes.
    """

    # init. vars.
    df = adj_mat.reset_index(drop=True)  # guarantee the indexes are 0, 1,...
    adjacency_matrix = df.values
    genes = df.columns
    # get W
    W_ = get_weight_matrix(adjacency_matrix)
    # p0 would have the value (1 / number of genes in anchor) where the anchor genes are, and 0 everywhere else
    p_0 = genes.isin(anchor_genes).astype(int) / len(anchor_genes)
    # propagate
    p_final, conv = propagate(adjacency_matrix,
                              p_0,
                              alpha=alpha_propagate,
                              tol=tol,
                              max_iter=max_iter,
                              W=W_)
    if verbose:
        print(f"Propagation vector converged after {conv} iterations")
    if normalize:
        p_norm, conv_norm = propagate(adjacency_matrix,
                                      p_0,
                                      alpha=alpha_normalize,
                                      tol=tol,
                                      max_iter=max_iter,
                                      W=W_)
        if np.count_nonzero(p_norm) != len(p_norm):  # zero divide
            p_norm[p_norm == 0] = np.inf
        p_final = p_final / p_norm
        if verbose:
            print(f"Normalize vector converged after {conv_norm} iterations")
    # return the final propagation vector
    return pd.Series(p_final, index=genes)


def get_adjacency_matrix(organism: str) -> pd.DataFrame:
    """
    For a given organism get the adjacency matrix based on PPI binary scores from BioGRID
    """

    # read PPI data
    if organism == "sce":  # S. cerevisiae
        path = cs.SCE_BIOGRID
        yeast = 'Saccharomyces cerevisiae (S288c)'
    else:  # S. pombe
        path = cs.SPO_BIOGRID
        yeast = 'Schizosaccharomyces pombe (972h)'
    df_interactions = pd.read_table(path, low_memory=False)
    df_ppi = df_interactions[df_interactions['Experimental System Type'] == 'physical']
    # we only want interactions within the organism
    df_ppi = df_ppi[(df_ppi['Organism Name Interactor A'] == yeast) & (df_ppi['Organism Name Interactor B'] == yeast)]
    df_ppi['ppi'] = 1
    # Create Graph
    G = nx.from_pandas_edgelist(df_ppi, source='Systematic Name Interactor A', target='Systematic Name Interactor B',
                                edge_attr='ppi')
    # Build PPI adjacency matrix
    return pd.DataFrame(nx.adjacency_matrix(G, weight='ppi').todense(), index=G.nodes, columns=G.nodes
                        ).reset_index().rename(columns={'index': 'gene'})


def orthologs_experiment_evaluation(y_test: np.ndarray,
                                    y_hat: np.ndarray,
                                    y_prob: np.ndarray) -> pd.Series:
    """
    Metrics for orthologs experiment evaluation
    """

    return pd.Series([roc_auc_score(y_test, y_prob), recall_score(y_test, y_hat), precision_score(y_test, y_hat)],
                     index=['AUC', 'Recall', 'Precision'])


def get_sce_to_spo_tlm_mapping() -> pd.DataFrame:
    """
    Return the mapping of TLM orthologs
    """

    # get the data
    pombe_cerev_df = get_spo_to_sce_orthologs()
    spo_tlms = get_spo_tlm()
    sce_tlms = get_sce_tlm()
    sce_tlms['tel_len'] = sce_tlms['tel_len'].apply(short_long)
    # build the mapping
    pombe_cerev_df['tlm_pombe'] = 'Non_pombe'
    pombe_cerev_df.loc[
        pombe_cerev_df['pombe'].isin(spo_tlms[spo_tlms['tel_len'] == 0]['gene']), 'tlm_pombe'] = 'Short_pombe'
    pombe_cerev_df.loc[
        pombe_cerev_df['pombe'].isin(spo_tlms[spo_tlms['tel_len'] == 1]['gene']), 'tlm_pombe'] = 'Long_pombe'
    pombe_cerev_df['tlm_cerev'] = 'Non_cerev'
    pombe_cerev_df.loc[
        pombe_cerev_df['cerev'].isin(sce_tlms[sce_tlms['tel_len'] == 0]['gene']), 'tlm_cerev'] = 'Short_cerev'
    pombe_cerev_df.loc[
        pombe_cerev_df['cerev'].isin(sce_tlms[sce_tlms['tel_len'] == 1]['gene']), 'tlm_cerev'] = 'Long_cerev'
    pombe_cerev_df['tlm_map'] = pombe_cerev_df['tlm_cerev'] + pombe_cerev_df['tlm_pombe']
    return pombe_cerev_df

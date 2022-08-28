"""
Constants, models are defined separately in models.py
"""

from typing import Final

# Experiments parameters (Seeds, folds etc.)
# LogisticRegressionCV random state for reproducibility
RANDOM_STATE_LRCV: Final[int] = 675
RANDOM_STATE_K_FOLDS: Final[int] = 1
RANDOM_STATE_PERMUTATION: Final[int] = 0
REPEATS: Final[int] = 5
FOLDS: Final[int] = 10
LABELS: tuple[str, str, str, str, str, str] = (
    "very short", "short", "slightly short", "slightly long", "long", "very long")
SPO_ANCHOR_GENES: tuple[str, ...] = ('SPAC16A10.07c', 'SPBC1778.02', 'SPAC19G12.13c', 'SPAC6F6.16c',
                                     'SPAC26H5.06', 'SPCC188.07', 'SPBC409.12c', 'SPCC1393.14',
                                     'SPBC2D10.13', 'SPBC29A3.14c', 'SPCC126.02c', 'SPBC543.03c')

# data
SCE_FULL_GI_SCORES: Final[str] = 'data/df_matrix_gi_all_genes.pkl'
SCE_KEGG_GENES: Final[str] = 'data/sce_KEGG_pathway_genes.pkl'
SPO_KEGG_GENES: Final[str] = 'data/spo_KEGG_pathway_genes.pkl'
SCE_KEGG_NAMES: Final[str] = 'data/sce_KEGG_pathway_names.pkl'
SPO_KEGG_NAMES: Final[str] = 'data/spo_KEGG_pathway_names.pkl'
SPO_POMBASE_COMPLEXES: Final[str] = 'data/spo_pombase_complexes.tsv'
SCE_CYC2008_COMPLEXES: Final[str] = 'data/sce_CYC2008_complexes.xls'
SCE_TLM_DATA: Final[str] = 'data/sce_TLM_data.xlsx'
SPO_TLM_DATA: Final[str] = 'data/spo_TLM_data.xlsx'
SPO_FYPO_DATA: Final[str] = "data/phenotype_annotations.pombase.phaf"
SPO_LIU_DATA: Final[str] = "data/spo_Liu_et_al_data.xlsx"
SPO_UNIPROT: Final[str] = "data/spo_uniprot.xlsx"
SCE_UNIPROT: Final[str] = "data/sce_uniprot.xlsx"
POMBASE_ORTHOLOGS: Final[str] = 'data/pombe_cerevisiae manually curated orthologs V2_23.txt'
SPO_TO_SCE_ORTHOLOGS: Final[str] = 'data/pombe_cerevisiae_orthologs_df.pkl'
SPO_BIOGRID: Final[str] = 'data/BIOGRID-ORGANISM-Schizosaccharomyces_pombe_972h-4.4.207.tab3.txt'
SCE_BIOGRID: Final[str] = 'data/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-4.4.207.tab3.txt'
SCE_GENES_YEASTMINE: Final[str] = 'data/sce_gene_descriptions_yeastmine.csv'
SPO_POMBASE_NAMES: Final[str] = 'data/spo_gene_IDs_names.tsv'
GO_DAG_BASIC: Final[str] = 'data/go-basic_release_2022_07_01.obo'
SCE_GAF: Final[str] = 'data/sgd_release_2022_07_01.gaf'

# features without target column, with 'gene' column and for ALL the genes
SCE_GO_BP_FEATURES: Final[str] = 'features/GO_BP_sce_features'
SCE_GO_CC_FEATURES: Final[str] = 'features/GO_CC_sce_features'
SCE_KEGG_FEATURES: Final[str] = 'features/KEGG_sce_features.pkl'
SPO_KEGG_FEATURES: Final[str] = 'features/KEGG_spo_features.pkl'
SCE_CYC2008_FEATURES: Final[str] = 'features/CYC2008_sce_features.pkl'
SPO_POMBASE_COMPLEX_FEATURES: Final[str] = 'features/POMBASE_spo_features.pkl'

# results of experiments
SINGLE_FEATURE_AUC: Final[str] = 'results/single_feature_auc.xlsx'
SINGLE_FEATURE_MCC: Final[str] = 'results/single_feature_mcc.xlsx'
PAIRWISE_FEATURES_AUC: Final[str] = 'results/pairwise_features_auc.xlsx'
PAIRWISE_FEATURES_MCC: Final[str] = 'results/pairwise_features_mcc.xlsx'
MULTI_CLASS_RESULTS: Final[str] = 'results/confusion_matrix_multi_class.xlsx'
SPO_TLM_SCORES: Final[str] = 'results/spo_short_vs_long_scores.pkl'
FEATURE_IMPORTANCE: Final[str] = 'results/spo_feature_importance.xlsx'
FEATURE_IMPORTANCE_COEF_NONZERO: Final[str] = 'results/spo_feature_importance_coef_nonzero.xlsx'
FEATURE_IMPORTANCE_FEATURES_NONZERO: Final[str] = 'results/spo_feature_importance_features_nonzero.pkl'
SPO_LOO_RESULTS: Final[str] = 'results/spo_leave_one_out_predictions.pkl'
TLM_VS_NON_RESULTS: Final[str] = 'results/tlm_vs_non_methods_results.pkl'

# paper tables
TABLE_1: Final[str] = 'tables/Table 1.xlsx'
TABLE_3: Final[str] = 'tables/Table 3.xlsx'
TABLE_4: Final[str] = 'tables/Table 4.xlsx'
TABLE_S1: Final[str] = 'tables/Table S1.xlsx'
TABLE_S2: Final[str] = 'tables/Table S2.xlsx'
TABLE_S4: Final[str] = 'tables/Table S4.xlsx'

# paper figures
FIGURE_1A_AUC: Final[str] = 'figures/Figure 1A AUC.png'
FIGURE_1A_MCC: Final[str] = 'figures/Figure 1A MCC.png'
FIGURE_1B_AUC: Final[str] = 'figures/Figure 1B AUC.png'
FIGURE_1B_MCC: Final[str] = 'figures/Figure 1B MCC.png'
FIGURE_2A_AUC: Final[str] = 'figures/Figure 2A AUC.png'
FIGURE_2A_MCC: Final[str] = 'figures/Figure 2A MCC.png'
FIGURE_2B_1: Final[str] = 'figures/Figure 2B scatter.png'
FIGURE_2B_2: Final[str] = 'figures/Figure 2B scatter enlarged.png'
FIGURE_2C_1: Final[str] = 'figures/Figure 2C samples used.png'
FIGURE_2C_2: Final[str] = 'figures/Figure 2C boxplot.png'
FIGURE_S1: Final[str] = 'figures/Figure of various telomere lengths Accuracy.png'
FIGURE_S2: Final[str] = 'figures/Figure of various telomere lengths F1.png'
FIGURE_S3: Final[str] = 'figures/Figure of various telomere lengths Recall.png'
FIGURE_3A: Final[str] = 'figures/Figure 3A AUC.png'
FIGURE_3B_1: Final[str] = 'figures/Figure 3B feature importance 1.png'
FIGURE_3B_2: Final[str] = 'figures/Figure 3B feature importance 2.png'

# miscellaneous
BOLD: Final[str] = '\033[1m'
END: Final[str] = '\033[0m'

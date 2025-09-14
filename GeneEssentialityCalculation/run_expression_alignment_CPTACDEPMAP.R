#### Part of the codes are from Celligner paper
#### Warren, A., Chen, Y., Jones, A. et al. Global computational alignment of 
#### tumor and cell line transcriptional profiles. Nat Commun 12, 22 (2021). 
#### https://doi.org/10.1038/s41467-020-20294-x

### This code is tested under Seurat 4.4.0. Please uncomment line 121 in src/Celligner_methods.R if you are using Seurat v5 or later
#设置禁用线程的配置参数：
rm(list = ls())
setwd("/yourworkspace/")
source("src/Celligner_methods.R") ### Please restart R if you see error this line

### This code is tested under Seurat 4.4.0. Please uncomment line 121 in src/Celligner_methods.R if you are using Seurat v5 or later
#设置禁用线程的配置参数：
configure.args <- c(preprocessCore = "--disable-threading")
options(configure.args = configure.args)
#重新安装 preprocessCore 包：
#BiocManager::install("preprocessCore")

datapath="./" #set to the path to TDtool
#setwd(datapath)

########Read CPTAC data, DepMap data and metadata#######
#### CCLE expression file CCLE_expression_full.csv is downloaded from depmap portal
#### All the other data are downloaded from https://github.com/broadinstitute/Celligner_ms
#tmp <- fread("./data/TumorCompendium_v10_PolyA_hugo_log2tpm_58581genes_2019-07-25.tsv")
#head(tmp[,1:6])
#tmp1 <- fread("./data/pancan_merged_CPTAC_RNAseq_Tumor.tsv")
#head(tmp1[,1:6])
dat = load_data(datapath,tumor_file = "./data/pancan_merged_CPTAC_RNAseq_Tumor.tsv", 
                cell_line_file = "./data/CCLE_expression_full.csv", hgnc_file = "./hgnc_complete_set_7.24.2018.txt")

common_DepMap_ID = intersect(rownames(dat$CCLE_mat),dat$CCLE_ann$sampleID)
dat$CCLE_mat = dat$CCLE_mat[common_DepMap_ID,]
dat$CCLE_ann = dat$CCLE_ann[match(common_DepMap_ID,dat$CCLE_ann$sampleID),]

####quantile norm####
library(preprocessCore)
comb_mat = cbind(t(dat$CCLE_mat),t(dat$tissue_mat))
comb_mat_norm = normalize.quantiles(comb_mat)
rownames(comb_mat_norm) = rownames(comb_mat)
colnames(comb_mat_norm) = colnames(comb_mat)

dat$CCLE_mat = t(comb_mat_norm[,rownames(dat$CCLE_mat)])
dat$tissue_mat = t(comb_mat_norm[,rownames(dat$tissue_mat)])
#####################

gene_stats <- calc_gene_stats(dat, datapath)

comb_ann <- rbind(
  dat$tissue_ann %>% dplyr::select(sampleID, lineage) %>%
    dplyr::mutate(type = 'tumor'),
  dat$CCLE_ann %>% dplyr::select(sampleID, lineage) %>%
    dplyr::mutate(type = 'CL')
)

########initial clustering for CPTAC and DepMap data#######

tissue_obj <- create_Seurat_object(dat$tissue_mat, dat$tissue_ann, type='tumor')
CCLE_obj <- create_Seurat_object(dat$CCLE_mat, dat$CCLE_ann, type='CL')

tissue_obj <- cluster_data(tissue_obj)
CCLE_obj <- cluster_data(CCLE_obj)

#######identify cluster specific genes based on initial clustering#######
tumor_DE_genes <- find_differentially_expressed_genes(tissue_obj)
CL_DE_genes <- find_differentially_expressed_genes(CCLE_obj)

DE_genes <- full_join(tumor_DE_genes, CL_DE_genes, by = 'Gene', suffix = c('_tumor', '_CL')) %>%
  mutate(
    tumor_rank = dplyr::dense_rank(-gene_stat_tumor),
    CL_rank = dplyr::dense_rank(-gene_stat_CL),
    best_rank = pmin(tumor_rank, CL_rank, na.rm=T)) %>%
  dplyr::left_join(gene_stats, by = 'Gene')

# take genes that are ranked in the top 1000 from either dataset, used for finding mutual nearest neighbors
DE_gene_set <- DE_genes %>%
  dplyr::filter(best_rank < global$top_DE_genes_per) %>%
  .[['Gene']]


#######apply cPCA and regress cPCs out#######
cov_diff_eig <- run_cPCA(tissue_obj, CCLE_obj, global$fast_cPCA)

if(is.null(global$fast_cPCA)) {
  cur_vecs <- cov_diff_eig$vectors[, global$remove_cPCA_dims, drop = FALSE]
} else {
  cur_vecs <- cov_diff_eig$rotation[, global$remove_cPCA_dims, drop = FALSE]
}

rownames(cur_vecs) <- colnames(dat$tissue_mat)
tissue_cor <- resid(lm(t(dat$tissue_mat) ~ 0 + cur_vecs)) %>% t()
CCLE_cor <- resid(lm(t(dat$CCLE_mat) ~ 0 + cur_vecs)) %>% t()

#######apply MNN#######
mnn_res <- run_MNN(CCLE_cor, tissue_cor,  k1 = global$mnn_k_tumor, k2 = global$mnn_k_CL, ndist = global$mnn_ndist, 
                   subset_genes = DE_gene_set)

corrected_data = t(rbind(mnn_res$corrected, CCLE_cor)) ### output aligned data
saveRDS(corrected_data,file="./data/aligned_expression_DEPMAP.rds")

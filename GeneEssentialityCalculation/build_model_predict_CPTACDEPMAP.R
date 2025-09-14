### This script is used to build elastic-net model to predict gene essentiality scores
rm(list=ls())
datapath="./"
setwd("/yourworkspace/script/CPTAC_dependency/")

## preprocess depmap gene essentiality data available at depmap portal
gene_effect_21Q1 = read.delim("./data/Achilles_gene_effect.csv",sep=",",header=T) ### gene effect file available at DEPMAP portal
screen_genes = sapply(strsplit(colnames(gene_effect_21Q1),"[..]"),"[[",1)[-1]
screen_cells = gene_effect_21Q1$DepMap_ID

gene_effect_21Q1 = t(as.matrix(gene_effect_21Q1[,-1]))
rownames(gene_effect_21Q1) = screen_genes
colnames(gene_effect_21Q1) = screen_cells

## get the list of models to train
cv_validated_models = read.delim("src/cv_validated_genelist.txt",header = F)$V1
gene_effect_21Q1 = gene_effect_21Q1[cv_validated_models,]

## preprocess expression data
corrected_data = readRDS("./data/aligned_expression_DEPMAP.rds")

## convert ensemble gene ID to gene official names
gene_mapping = read.delim("src/ensemble_mapping.txt")
genenames_real = gene_mapping$Gene.name[match(rownames(corrected_data),gene_mapping$Gene.stable.ID)]
rownames(corrected_data) = genenames_real

## process essentiality and expression data to have same cell lines
common_cells = intersect(colnames(gene_effect_21Q1),colnames(corrected_data))
#dim(corrected_data_tissue)
corrected_data_CCLE = corrected_data[,common_cells]
corrected_data_tissue = corrected_data[,grepl("Tumor",colnames(corrected_data))]
CCLE_expr_scaled = t(scale(t(corrected_data_CCLE)))
tissue_expr_log2TPM_scaled = t(scale(t(corrected_data_tissue)))

gene_effect_21Q1 = gene_effect_21Q1[,common_cells]

## impute the missing values by median
median_imp = function(x)
{
  x[is.na(x)] = median(x[!is.na(x)])
  return(x)
}

gene_effect_21Q1_imp = t(apply(gene_effect_21Q1,1,median_imp))

###train gene model and predict on CPTAC###
library(parallel)
library(glmnet)

Elastic_model = function(y,x)
{
  cvfit_gene = cv.glmnet(t(x),y,nfolds = 10,alpha=0.5)
  pred_score_gene = predict(cvfit_gene,newx = t(x),s="lambda.min")
  pred_ability = cor(pred_score_gene,y)
  res = list("model"=cvfit_gene,"predictability"=pred_ability)
}

## convert training data to list for multiprocess
para_data = lapply(rownames(gene_effect_21Q1_imp),function(x) gene_effect_21Q1_imp[x,])
names(para_data) = rownames(gene_effect_21Q1_imp)

## train the model
ptm = proc.time()
elastic_coef_all = mclapply(para_data,function(x) Elastic_model(x,CCLE_expr_scaled),mc.cores = 50) ### use 50 cores to train the model
print(proc.time()-ptm)

## predict on CPTAC
ptm=proc.time()
tissue_ess = mclapply(elastic_coef_all,function(x) predict(x$model,newx=t(tissue_expr_log2TPM_scaled),s="lambda.min")[,1],mc.cores = 50)
proc.time() - ptm

## generate CPTAC essentiality matrix
library(rlist)
tissue_ess_data = list.rbind(tissue_ess)
dim(tissue_ess_data)
row.names(tissue_ess_data)
#1966 1659
saveRDS(tissue_ess_data,file="./data/CPTAC_essentiality_scores.rds")

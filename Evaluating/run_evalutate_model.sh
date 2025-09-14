#!/bin/bash
# run_eval.sh

for ct in BLCA BRCA CESC COAD ESCA HNSC KIRC KIRP LGG LIHC LUSC PAAD PCPG PRAD SARC SKCM STAD TGCT THCA THYM UCEC LUAD; do
  for pt in tformer tformer_lin Agent MLP; do
    for em in vit resnet; do
      echo "[$(date '+%F %T')] Start: $ct $pt $em"
      python /backup/lgx/path_omics_t/script/evaluate_model.py \
             --cancer_type "$ct" \
             --prediction_type "$pt" \
             --extraction_model "$em"
      echo "[$(date '+%F %T')] Done : $ct $pt $em"
      echo "-------------------------------------------------------------"
    done
  done
done



#!/bin/bash
# save as clean_results.sh  &&  chmod +x clean_results.sh
BASE="/backup/lgx/path_omics_t/data/result/model"
# 外层循环：癌症类型
for ct in BLCA BRCA CESC COAD ESCA HNSC KIRC KIRP LGG LIHC LUSC PAAD PCPG PRAD SARC SKCM STAD TGCT THCA THYM UCEC LUAD; do
  # 中层循环：预测模型
  for HEAD in tformer tformer_lin Agent MLP; do
    # 内层循环：特征提取模型
    for MODEL in vit resnet; do
      DIR="${BASE}/${MODEL}/${HEAD}/${ct}/result"
      if [[ -d "$DIR" ]]; then
        echo "清理: $DIR"
        # 删除目录内所有内容
        rm -rf "${DIR:?}/"* 2>/dev/null
        # 如想连 result/ 目录一起删掉，把下一行取消注释
        # rmdir "$DIR" 2>/dev/null
      fi
    done
  done
done
echo "清理完成！"





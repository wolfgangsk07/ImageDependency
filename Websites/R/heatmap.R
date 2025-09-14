## 0. 环境 ---------------------------------------------------------
#.libPaths("../../R/libraries")
library(rjson)
getStrLen<-function(str,cex=1){
  
  lettersSize<-c(11.1,11.1,12,12,11.1,10.2,13,12,4.6,8.4,
                 11.1,9.3,13.9,12,13,11.1,13,12,11.1,10.2,
                 12,11.1,15.7,11.1,11.1,10.2,
                 9.3,9.1,8.4,9.3,9.3,4.6,9.3,9.3,3.7,3.7,8.4,3.7,13.9,9.3,9.3,9.3,9.3,5.5,8.4,4.6,9.3,8.4,12,8.3,8.3,8.3,
                 9.7,9.7,9.3,4.7,4.7,14.9,5.6,5.6,9.3,9.3,9.3,9.3,9.3,9.3,9.3,9.3,9.3,9.3,4.7,9.7)/100
  symbols<-c("-","+","_",",",".","%","(",")",0:9," ","?")
  text<-c(LETTERS,letters,symbols)
  names(lettersSize)<-text
  fullLengh<-0
  for(i in 1:nchar(str)){
    if(substring(str,i,i) %in% names(lettersSize)){
      fullLengh<-lettersSize[substring(str,i,i)]+fullLengh
    }else{
      fullLengh<-mean(lettersSize)+fullLengh
      print(paste(substring(str,i,i)," is unknown, replaced with average length!"))
    }
  }
  return(as.numeric(fullLengh*cex))
  
}
getResult<-function(task_id){
	library("RSQLite")
	genenames<-readLines("../../R/outputgenes.txt")
	TCGAcenter<-read.table("../../R/TCGAcenter.txt")
	# 2. 加载包
	library(DBI)
	# 3. 建立数据库连接（假设数据库路径为'database/database.db'）
	con <- dbConnect(RSQLite::SQLite(), "../../database/database.db")

	# 4. 安全执行查询（防止SQL注入）
	query <- "SELECT result_json FROM analysis_tasks WHERE task_id = ?"
	result <- dbGetQuery(con, query, params = list(task_id))[1,1]
	
	#query <- "SELECT * FROM uploads WHERE taskuuid = ?"
	#files <- dbGetQuery(con, query, params = list(task_id))
	# 5. 关闭连接
	dbDisconnect(con)
	result_list<-fromJSON(result)
	#write(toJSON(result_list),paste0("../../results/",task_id,"/exprs.json"))
	outputlist<-list()
	errors<-c()
	outputtab<-data.frame(matrix(ncol = 1966, nrow = 0))
	colnames(outputtab)<-genenames
	for(i in 1:length(result_list)){
		if(is.numeric(result_list[[i]]$result)){
			outputtab[result_list[[i]]$filename,] <- result_list[[i]]$result
			# 数据标准化
			  scaled_result <- as.numeric(
				(result_list[[i]]$result - TCGAcenter[1, ]) / TCGAcenter[2, ]
			  )
			  names(scaled_result) <- genenames
			  
			  # 关键修复：同时添加result和filename到同一个列表元素
			  outputlist[[length(outputlist) + 1]] <- list(
				result = scaled_result,
				filename = result_list[[i]]$filename
			  )
			
		}else{
			errors<-c(errors,paste0(result_list[[i]]$filename,":",result_list[[i]]$result$error))
		}
	}
	if(length(errors)>0){
		writeLines(errors,paste0("../../results/",task_id,"/errors.txt"))
	}
	
	write.table(outputtab,paste0("../../results/",task_id,"/rawexprs.txt"),quote=FALSE,sep="\t")
	return(outputlist)

}
drawHeatmap<-function(task_id,cancer_type){
	# 1. 安装包（如未安装）

	result_list<-getResult(task_id)
	library(ComplexHeatmap)
	library(RColorBrewer)

	## 1. 自定义颜色函数（Nature/NEJM 级） -----------------------------
	pal_subtype <- function(n) {
	  if (n == 1)  return("#0072B2")
	  if (n == 2)  return(c("#0072B2", "#E79F00"))
	  if (n <= 8)  return(brewer.pal(n, "Dark2"))
	  viridis_pal(option = "A", begin = .15, end = .85)(n)
	}

	CPTAC_druggable <- read.delim("../../R/CPTAC_druggable.txt",check.names = FALSE)


	## 3. 预存热图列表 --------------------------------------------------

	topannos<-list()
	mats<-list()
	



	## 3.1 读 subtype -------------------------------------------------
	path_txt   <- paste0("../../R/TCGAdep_cluster/",cancer_type,"_subtype.txt")
	TCGASubtype <- read.delim(path_txt, stringsAsFactors = FALSE)

	commongenes<-intersect(colnames(TCGASubtype)[-c(1,2)],names(result_list[[1]]$result))
	#确定每个svs样本的分类
	subtypeIdentify<-c()
	for(i in 1:length(result_list)){
		predictGenes<-result_list[[i]]$result[commongenes]
		cors<-c()
		for(subtype in unique(TCGASubtype[,2])){
			subset<-TCGASubtype[TCGASubtype[,2]==subtype,-c(1, 2)]
			cors[subtype]<-cor(colMeans(subset),predictGenes)
			
		
		}
		subtypeIdentify[i]<-names(cors)[cors==max(cors)]

	}

	for(i in 1:length(result_list)){
	
		predictGenes<-result_list[[i]]$result[commongenes]
		toInsert<-tail(which(TCGASubtype[,2]==subtypeIdentify[i]),1)
		newRow <- data.frame(
			filename = result_list[[i]]$filename,  # 保持字符型
			subtype = subtypeIdentify[i],              # 保持与原列相同类型
			as.list(predictGenes),                     # 数值向量转为列表以匹配列数
			stringsAsFactors = FALSE                   # 避免因子转换
		)
		colnames(newRow) <- colnames(TCGASubtype)
		TCGASubtype<-rbind(head(TCGASubtype,toInsert),newRow,tail(TCGASubtype,nrow(TCGASubtype)-toInsert))
	}
	
	#获取svs样本位置
	
	SVSindex<-which(TCGASubtype[,1] %in% sapply(result_list,function(x)x$filename))
	print(SVSindex)
	ht <- Heatmap(
	t(TCGASubtype[, -c(1, 2)]),
	cluster_columns        = FALSE,
	cluster_rows           = TRUE,
	show_row_names         = FALSE,
	show_column_names      = FALSE,
	show_row_dend          = FALSE,
	use_raster             = FALSE
	)
	pdf(NULL)
	draw(ht)
	mat<-as.matrix(ht@matrix)[row_order(ht),]
	topAnno<-table(TCGASubtype$Subtype_mRNA)
	
	dev.off()
	
	
	#开始作图
	svsfilenameHeight<-max(sapply(TCGASubtype[SVSindex,1],getStrLen,0.5))/2^0.5
	subtypeHeight<-max(sapply(names(topAnno),getStrLen,0.75))/2^0.5
	
	dpi<-300
	SVSCellWidth<-0.1*dpi
	top<-0.2*dpi+subtypeHeight*dpi
	bottom<-0.2*dpi+svsfilenameHeight*dpi
	width<-7*dpi+SVSCellWidth*length(SVSindex)
	height<-5*dpi+bottom+top
	left<-0.8*dpi
	right<-1*dpi
	
	
	group_gap<-0.1*dpi
	gap<-0.05*dpi

	topAnnoAreaHeight<-0.2*dpi
	topAnnoHeight<-0.15*dpi
	library(RColorBrewer)
	FCGradiant<-colorRampPalette(c("#3B5F8A", "white", "#C23B22"))(20)
	lim<-c(-2,2)
	dir.create(paste0("../../results/",task_id),recursive=TRUE,showWarnings=FALSE)
	for(d in 1:2){
		if(d ==1){
			pdf(paste0("../../results/",task_id,"/heatmap.pdf"),width=width/dpi,height=height/dpi)
		}else{
			jpeg(paste0("../../results/",task_id,"/heatmap.jpg"), width=width/dpi,height=height/dpi, units="in",res=300, quality=25)
		}
		par(mai=c(0,0,0,0),omi=c(0,0,0,0))
		plot(x=-1,y=-1,xlim=c(0,width),ylim=c(0,height),axes=FALSE,xaxs="i",yaxs="i",col="transparent")


		Hline<-0.3*dpi



		cancerName<-cancer_type


		subLeft<-left
		subtop<-top
		subBottom<-bottom

		subHeight<-height-top-bottom

		cellHeight<-(subHeight-topAnnoAreaHeight)/nrow(mat)
		
		cellWidth<-(width-left-right-SVSCellWidth*length(SVSindex))/(ncol(mat)-length(SVSindex))
		


		print(dim(mat))
		#画heatmap
		for(j in 1:ncol(mat)){
			if(j %in% SVSindex){
				currCellWidth<-SVSCellWidth
			}else{
				currCellWidth<-cellWidth
			}
			for(i in 1:nrow(mat)){
			
				rectcol<-FCGradiant[round((mat[i,j]-lim[1])/(lim[2]-lim[1])*20,0)]
				
				cellLeft<-subLeft+(j-1)*cellWidth+sum(j>SVSindex)*(SVSCellWidth-cellWidth)
				cellBottom<-subBottom+subHeight-topAnnoAreaHeight-i*cellHeight
				
				rect(cellLeft,cellBottom,cellLeft+currCellWidth,cellBottom+cellHeight,border=NA,col=rectcol)
			}
			if(j %in% SVSindex){
				rect(cellLeft,subBottom,cellLeft+currCellWidth,subBottom+subHeight-topAnnoAreaHeight,border="white",col="transparent")
				points(rep(cellLeft+currCellWidth/2,2),subBottom-c(0,0.1*dpi),type="l")
				text(TCGASubtype[j,1],x=cellLeft+currCellWidth/2,y=subBottom-0.1*dpi,adj=c(1,1),srt=45,cex=0.5)
			}
		}

		#标注genename
		geneMarkCount<-sum(rownames(mat) %in% CPTAC_druggable $`Gene Symbol`)
		fontsize=0.3+0.5*(60-geneMarkCount)/60
		geneMarkHeight<-(subHeight-topAnnoAreaHeight)/geneMarkCount
		geneMarkBaseline<-subBottom+subHeight-topAnnoAreaHeight
		geneMarkRight<-subLeft-0.2*dpi
		for(i in 1:nrow(mat)){
			if(rownames(mat)[i] %in% CPTAC_druggable $`Gene Symbol`){
				text(rownames(mat)[i],x=geneMarkRight,y=geneMarkBaseline-geneMarkHeight/2,cex=fontsize,adj=c(1,0.5))
				linesLeftEndY<-geneMarkBaseline-geneMarkHeight/2
				linesRightEndY<-subBottom+subHeight-topAnnoAreaHeight-(i-0.5)*cellHeight
				
				points(geneMarkRight+(subLeft-geneMarkRight)*c(0,0.3,0.7,1),linesLeftEndY+(linesRightEndY-linesLeftEndY)*c(0,0,1,1),type="l",lwd=0.5+0.5*(60-geneMarkCount)/60)
				geneMarkBaseline<-geneMarkBaseline-geneMarkHeight
			}
		}

		#画topanno
		topannoLeft<-subLeft
		for(i in 1:length(topAnno)){
			SVScount<-sum(TCGASubtype[SVSindex,2]==names(topAnno)[i])
			topannoLen<-(topAnno[i]-SVScount)*cellWidth+SVScount*SVSCellWidth
			rect(topannoLeft,subBottom+subHeight-topAnnoHeight,topannoLeft+topannoLen,subBottom+subHeight,border=NA,col=pal_subtype(length(topAnno))[i])
			tag<-names(topAnno)[i]

			text(paste0(" ",tag),x=topannoLeft+topannoLen*0.3,y=subBottom+subHeight-topAnnoAreaHeight+topAnnoHeight*1.5,adj=c(0,0),cex=0.75,srt=30,font=2)
			topannoLeft<-topannoLeft+topannoLen

		}




		#colorbar
		legendWidth<-0.2*dpi
		legendHeight<-1.5*dpi

		legendLeft<-width-right+gap*3
		legendBottom<-height-top-topAnnoAreaHeight-legendHeight

		colCellHeight<-legendHeight/length(FCGradiant)
		for(i in 1:length(FCGradiant)){
			colCellBottom<-legendBottom+(i-1)*colCellHeight
			rect(legendLeft,colCellBottom,legendLeft+legendWidth,colCellBottom+colCellHeight,border=NA,col=FCGradiant[i])
		}


		text(lim[1],x=legendLeft+legendWidth+0.05*dpi,y=legendBottom,adj=c(0,0),cex=0.75,font=2)
		text(lim[2],x=legendLeft+legendWidth+0.05*dpi,y=legendBottom+legendHeight,adj=c(0,1),cex=0.75,font=2)
		text(0,x=legendLeft+legendWidth+0.05*dpi,y=legendBottom+legendHeight/2,adj=c(0,0.5),cex=0.75,font=2)
		text("Essentiality",x=legendLeft,y=legendBottom+legendHeight+0.05*dpi,font=2,adj=c(0,0),cex=0.75)
		
		dev.off()
	}
	write.table(TCGASubtype,paste0("../../results/",task_id,"/exprs.txt"),quote=FALSE,sep="\t",row.names=FALSE)
}

args <- commandArgs(trailingOnly = TRUE)
task_id<-args[1]
cancerName<-args[2]
if(is.null(task_id)){
	task_id<-"5dc7614b-3760-4f8f-a230-f23bdd04c197"
	cancerName<-"LUAD"
}
drawHeatmap(task_id,cancerName)

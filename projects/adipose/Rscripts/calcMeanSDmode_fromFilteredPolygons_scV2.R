

stats<-read.csv("/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/finalResults/multiRun_Total_from1_to927V2.csv",as.is=T)

raw_sc<-read.csv("/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/polygonsFiltered_sc_from1_to950V3.txt",as.is=T,h=F,fill=T,col.names=paste0("V",seq_len(47926)))

colnames(raw_sc)[1:4]<-c("cohort","filename","N","below750")

## check that there are no lines split across two rows (because more polygons than 47926 - cols in df)
print("this many first column values are numeric")
print(table(!is.na(as.numeric(raw_sc[,1]))))


table(stats$filename%in%c(sapply(raw_sc$filename,basename)))

head(stats[ !stats$filename%in%c(sapply(raw_sc$filename,basename)),])

raw_sc<-raw_sc[ !grepl("Visceral",raw_sc$filename) & !grepl("[0-9]vc[0-9]",raw_sc$filename),]
raw_sc<-raw_sc[ raw_sc$below750<=0.9,]
raw_sc<-raw_sc[ !grepl("faultySlides",raw_sc$filename),]


ID<-sapply(sapply(sapply(sapply(raw_sc$filename,basename),function(x) unlist(strsplit(x,"\\."))[1]),function(x) sub("vc[0-9]","vc",x)),function(x) sub("sc[0-9]","sc",x))

## mistake is that some slides have been run more than once, because they failed in the initial run


raw_sc$cohort<-ID
colnames(raw_sc)[1]<-"ID"



mean_list<-list()
sd_list<-list()

# calculates smooth histogram and grabs the value with highest density
emilMode<-function(x){
	if(sum(!is.na(x))>1){
		den<-density(x,na.rm=T)
		return(den$x[which.max(den$y)])
	} else{
	  return(NA)
	}
}

mode_list<-list()

for(id in unique(raw_sc$ID)){       
       mean_list[[paste0(id)]]<-mean(unlist(raw_sc[ raw_sc$ID%in%id,5:ncol(raw_sc)]),na.rm=T)
       sd_list[[paste0(id)]]<-sd(unlist(raw_sc[ raw_sc$ID%in%id,5:ncol(raw_sc)]),na.rm=T)
       mode_list[[paste0(id)]]<-emilMode(unlist(raw_sc[ raw_sc$ID%in%id,5:ncol(raw_sc)]))
}



df<-data.frame(ID=names(mean_list),mean=unlist(mean_list),sd=unlist(sd_list),mode=unlist(mode_list))

write.csv(df,"meanSDmode_from_polygonsFiltered_sc_from1_to950_V2.csv",col.names=T,row.names=F,qu=F)



stats<-read.csv("/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/finalResults/multiRun_Total_from1_to927V2.csv",as.is=T)

raw_vc<-read.csv("/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/polygonsFiltered_forHistPerIndividualPlot/polygonsFiltered_vc_from1_to950V3.txt",as.is=T,h=F,fill=T,col.names=paste0("V",seq_len(61248)))

colnames(raw_vc)[1:4]<-c("cohort","filename","Nadipocytes","below750")

## check that there are no lines split across two rows (because more polygons than 61248 - cols in df)
print("this many first column values are numeric")
print(table(!is.na(as.numeric(raw_vc[,1]))))


table(stats$filename%in%c(sapply(raw_vc$filename,basename)))

head(stats[ !stats$filename%in%c(sapply(raw_vc$filename,basename)),])

raw_vc<-raw_vc[ !grepl("Subcutaneous",raw_vc$filename) & !grepl("[0-9]sc[0-9]",raw_vc$filename),]
raw_vc<-raw_vc[ !(raw_vc$below750>0.65 & raw_vc$Nadipocytes<1500),]
raw_vc<-raw_vc[ !grepl("faultySlides",raw_vc$filename),]


ID<-sapply(sapply(sapply(sapply(raw_vc$filename,basename),function(x) unlist(strsplit(x,"\\."))[1]),function(x) sub("vc[0-9]","vc",x)),function(x) sub("sc[0-9]","sc",x))

## mistake is that some slides have been run more than once, because they failed in the initial run


raw_vc$cohort<-ID
colnames(raw_vc)[1]<-"ID"



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

print("This many WSIs")
print(length(unique(raw_vc$ID)))
count<-1

for(id in unique(raw_vc$ID)){       
       mean_list[[paste0(id)]]<-mean(unlist(raw_vc[ raw_vc$ID%in%id,5:ncol(raw_vc)]),na.rm=T)
       sd_list[[paste0(id)]]<-sd(unlist(raw_vc[ raw_vc$ID%in%id,5:ncol(raw_vc)]),na.rm=T)
       mode_list[[paste0(id)]]<-emilMode(unlist(raw_vc[ raw_vc$ID%in%id,5:ncol(raw_vc)]))
       print(count)
       count<-count+1
}



df<-data.frame(ID=names(mean_list),mean=unlist(mean_list),sd=unlist(sd_list),mode=unlist(mode_list))

write.csv(df,"meanSDmode_from_polygonsFiltered_vc_from1_to950_V3.csv",col.names=T,row.names=F,qu=F)

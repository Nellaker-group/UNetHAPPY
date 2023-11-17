

csv1<-read.csv("task_id2slide_id_31mar_shuffledV2.csv",as.is=T,header=F)
csv2<-read.csv("task_id2slide_id_17apr_shuffledV2restOfSlides.csv",as.is=T,header=F)

csv<-data.frame(rbind(csv1,csv2),stringsAsFactors=FALSE)

#copy and paste table from sqlite3 "
("(base) swf744@:/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/db$ sqlite3 main.db")
("select * from slide;")
slides<-read.table("SELECT_ALL_FROM_SLIDE",sep="|",header=F,as.is=T)

csv$path <- ""

index=1
for(i in csv$V2){
      csv[ index,"path"]<-slides[ slides$V1==i,"V2"]
      index<-index+1
}


colnames(csv)<-c("task_id","slide_id","path")
write.csv(csv,"task_id2slide_id2path.csv",col.names=T,qu=F,row.names=F)
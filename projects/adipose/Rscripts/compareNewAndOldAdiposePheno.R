
oldPhe<-read.csv("/gpfs3/well/lindgren/users/swf744/adipocyte/data/raw/mergedPhenotypeFile.csv",as.is=T)

newPheSC<-read.csv("/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/meanSDmode_from_polygonsFiltered_sc_from1_to1000_FIX.csv",as.is=T)
newPheSC<-read.csv("/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/meanSDmode_from_polygonsFiltered_vc_from1_to1000_FIX.csv",as.is=T)

newPheSC$ID2<-sapply(newPheSC$ID,function(x) sub("sc","",x))
newPheSC$ID2<-sapply(newPheSC$ID2,function(x) sub("vc","",x))
newPheSC$ID2<-sapply(newPheSC$ID2,function(x) unlist(strsplit(x,"_"))[1])

newPheVC$ID2<-sapply(newPheVC$ID,function(x) sub("sc","",x))
newPheVC$ID2<-sapply(newPheVC$ID2,function(x) sub("vc","",x))
newPheVC$ID2<-sapply(newPheVC$ID2,function(x) unlist(strsplit(x,"_"))[1])


table(unique(c(newPheSC$ID2,newPheVC$ID2))%in%oldPhe$ID)


head(oldPhe[ !oldPhe$ID%in%c(newPheSC$ID2,newPheVC$ID2),])
tail(oldPhe[ !oldPhe$ID%in%c(newPheSC$ID2,newPheVC$ID2),])
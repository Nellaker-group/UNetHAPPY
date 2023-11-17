setwd("/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/finalResults/")

old_sc<-read.csv("meanSDmode_from_polygonsFiltered_sc_from1_to1000_FIX.csv",as.is=T)
old_vc<-read.csv("meanSDmode_from_polygonsFiltered_vc_from1_to1000_FIX.csv",as.is=T)
new_sc<-read.csv("meanSDmode_from_polygonsFiltered_sc_from1_to950_V3.csv",as.is=T)
new_vc<-read.csv("meanSDmode_from_polygonsFiltered_vc_from1_to950_V3.csv",as.is=T)

int_vc<-intersect(old_vc$ID,new_vc$ID)
int_sc<-intersect(old_sc$ID,new_sc$ID)


old_sc<-old_sc[ old_sc$ID%in%int_sc,]
new_sc<-new_sc[ new_sc$ID%in%int_sc,]
old_sc<-old_sc[ order(match(old_sc$ID, new_sc$ID)),]
table(old_sc$ID==new_sc$ID)


old_vc<-old_vc[ old_vc$ID%in%int_vc,]
new_vc<-new_vc[ new_vc$ID%in%int_vc,]
old_vc<-old_vc[ order(match(old_vc$ID, new_vc$ID)),]
table(old_vc$ID==new_vc$ID)

par(mfrow=c(1,2))
plot(old_sc$mean,new_sc$mean,xlab="Mean size - SC - previous filtering",ylab="Mean size - SC -new filtering",main=paste0("R2=",cor(old_sc$mean,new_sc$mean,use="pairwise.complete.obs")**2)); abline(a=0,b=1)
plot(old_vc$mean,new_vc$mean,xlab="Mean size - VC - previous filtering",ylab="Mean size - VC -new filtering",main=paste0("R2=",cor(old_vc$mean,new_vc$mean,use="pairwise.complete.obs")**2)); abline(a=0,b=1)

plot(old_vc$sd,new_vc$sd,xlab="SD - VC - previous filtering",ylab="SD - VC -new filtering",main=paste0("R2=",cor(old_vc$sd,new_vc$sd,use="pairwise.complete.obs")**2)); abline(a=0,b=1)
plot(old_sc$sd,new_sc$sd,xlab="SD - SC - previous filtering",ylab="SD - SC -new filtering",main=paste0("R2=",cor(old_sc$sd,new_sc$sd,use="pairwise.complete.obs")**2)); abline(a=0,b=1)

plot(old_sc$mode,new_sc$mode,xlab="Mode - SC - previous filtering",ylab="Mode - SC -new filtering",main=paste0("R2=",cor(old_sc$mode,new_sc$mode,use="pairwise.complete.obs")**2)); abline(a=0,b=1)
plot(old_vc$mode,new_vc$mode,xlab="Mode - VC - previous filtering",ylab="Mode - VC -new filtering",main=paste0("R2=",cor(old_vc$mode,new_vc$mode,use="pairwise.complete.obs")**2)); abline(a=0,b=1)

##############################################
###############################################

csv_old<-read.csv("../polygonsFiltered_forHistPerIndividualPlot/polygonsFiltered_vc_from1_to950V3.txt",h=F,nrows=100,col.names=paste0("V",seq_len(47926)))
csv_new<-read.csv("../polygonsFiltered_forHistPerIndividualPlot/polygonsFiltered_vc_from1_to1000.txt",h=F,nrows=100,col.names=paste0("V",seq_len(47926)))

par(mfrow=c(1,2))
whichRow<-8; hist(unlist(csv_old[whichRow,5:ncol(csv_old)]),breaks=100,xlim=c(0,25000),ylim=c(0,750),main=paste0("mean=",mean(unlist(csv_old[whichRow,5:ncol(csv_old)]),na.rm=T)))
abline(v=mean(unlist(csv_old[whichRow,5:ncol(csv_old)]),na.rm=T),lwd=2) 
hist(unlist(csv_new[whichRow,5:ncol(csv_new)]),breaks=100,xlim=c(0,25000),ylim=c(0,750),main=paste0("mean=",mean(unlist(csv_new[whichRow,5:ncol(csv_new)]),na.rm=T)))
abline(v=mean(unlist(csv_new[whichRow,5:ncol(csv_new)]),na.rm=T),lwd=2)




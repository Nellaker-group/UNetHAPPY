
sc<-read.csv("finalResults/meanSDmode_from_polygonsFiltered_sc_from1_to1000_FIX.csv",as.is=T)
vc<-read.csv("finalResults/meanSDmode_from_polygonsFiltered_vc_from1_to1000_FIX.csv",as.is=T)



endoxID <- read.csv("/gpfs3/well/lindgren/craig/isbi-2012/NDOG_histology_IDs_and_quality.csv",as.is=T)
endoxID$filename2<-sub("\\.scn","",endoxID$filename)

endoxID_sc<-endoxID[ endoxID$depot%in%"subcutaneous",]
endoxID_vc<-endoxID[ endoxID$depot%in%"visceral",]


png("plots/hist_sc_modePerCohort.png",width=10,height=7,units="in",res=200)
par(mfrow=c(2,3))
hist(sc[ ,"mode"],breaks=100,main="ALL",ylim=c(0,0.0025),xlim=c(0,10000),freq=FALSE)
hist(sc[ grepl("^a",sc$ID),"mode"],breaks=100,main="Leipzig",ylim=c(0,0.0025),xlim=c(0,10000),freq=FALSE)
hist(sc[ grepl("^m",sc$ID),"mode"],breaks=100,main="Munich",ylim=c(0,0.0025),xlim=c(0,10000),freq=FALSE)
hist(sc[ grepl("^h",sc$ID),"mode"],breaks=100,main="Hohenheim",freq=FALSE)
hist(sc[ grepl("GTEX",sc$ID),"mode"],breaks=100,main="GTEX",ylim=c(0,0.0025),xlim=c(0,10000),freq=FALSE)
hist(vc[ vc$ID%in%endoxID_vc$filename2,"mode"],breaks=100,main="Endox",ylim=c(0,0.0025),xlim=c(0,10000),freq=FALSE)

dev.off()

png("plots/hist_vc_modePerCohort.png",width=10,height=7,units="in",res=200)
par(mfrow=c(2,3))
hist(vc[ ,"mode"],breaks=100,main="ALL",ylim=c(0,0.0025),xlim=c(0,10000),freq=FALSE)
hist(vc[ grepl("^a",vc$ID),"mode"],breaks=100,main="Leipzig",ylim=c(0,0.0025),xlim=c(0,10000),freq=FALSE)
hist(vc[ grepl("^m",vc$ID),"mode"],breaks=100,main="Munich",ylim=c(0,0.0025),xlim=c(0,10000),freq=FALSE)
hist(vc[ grepl("^h",vc$ID),"mode"],breaks=100,main="Hohenheim",freq=FALSE)
hist(vc[ grepl("GTEX",vc$ID),"mode"],breaks=100,main="GTEX",ylim=c(0,0.0025),xlim=c(0,10000),freq=FALSE)
hist(vc[ vc$ID%in%endoxID_sc$filename2,"mode"],breaks=100,main="Endox",ylim=c(0,0.0025),xlim=c(0,10000),freq=FALSE)
dev.off()

png("plots/cor_modeAndMean.png",width=10,height=7,units="in",res=200)
par(mfrow=c(1,2))
plot(vc[ ,"mean"],vc[ ,"mode"],xlab="mean",ylab="mode",main="VC - ALL")
abline(b=1,a=0)
plot(sc[ ,"mean"],sc[ ,"mode"],xlab="mean",ylab="mode",main="SC - ALL")
abline(b=1,a=0)
dev.off()

png("plots/cor_vc_perCohortmodeAndMean.png",width=10,height=7,units="in",res=200)
par(mfrow=c(2,3))
plot(vc[ ,"mean"],vc[ ,"mode"],xlab="mean",ylab="mode",main="VC - ALL")
abline(b=1,a=0)
plot(vc[ grepl("^a",vc$ID),"mean"],vc[ grepl("^a",vc$ID),"mode"],xlab="mean",ylab="mode",main="VC - Leipzig")
abline(b=1,a=0)
plot(vc[ grepl("^m",vc$ID),"mean"],vc[ grepl("^m",vc$ID),"mode"],xlab="mean",ylab="mode",main="VC - Munich")
abline(b=1,a=0)
plot(vc[ grepl("^h",vc$ID),"mean"],vc[ grepl("^h",vc$ID),"mode"],xlab="mean",ylab="mode",main="VC - Hohenheim")
abline(b=1,a=0)
plot(vc[ grepl("GTEX",vc$ID),"mean"],vc[ grepl("GTEX",vc$ID),"mode"],xlab="mean",ylab="mode",main="VC - GTEX")
abline(b=1,a=0)
plot(vc[ vc$ID%in%endoxID_vc$filename2,"mean"],vc[ vc$ID%in%endoxID_vc$filename2,"mode"],xlab="mean",ylab="mode",main="VC - Endox")
abline(b=1,a=0)

dev.off()

png("plots/cor_sc_perCohortmodeAndMean.png",width=10,height=7,units="in",res=200)
par(mfrow=c(2,3))
plot(sc[ ,"mean"],sc[ ,"mode"],xlab="mean",ylab="mode",main="SC - ALL")
abline(b=1,a=0)
plot(sc[ grepl("^a",sc$ID),"mean"],sc[ grepl("^a",sc$ID),"mode"],xlab="mean",ylab="mode",main="SC - Leipzig")
abline(b=1,a=0)
plot(sc[ grepl("^m",sc$ID),"mean"],sc[ grepl("^m",sc$ID),"mode"],xlab="mean",ylab="mode",main="SC - Munich")
abline(b=1,a=0)
plot(sc[ grepl("^h",sc$ID),"mean"],sc[ grepl("^h",sc$ID),"mode"],xlab="mean",ylab="mode",main="SC - Hohenheim")
abline(b=1,a=0)
plot(sc[ grepl("GTEX",sc$ID),"mean"],sc[ grepl("GTEX",sc$ID),"mode"],xlab="mean",ylab="mode",main="SC - GTEX")
abline(b=1,a=0)
plot(vc[ vc$ID%in%endoxID_sc$filename2,"mean"],vc[ vc$ID%in%endoxID_sc$filename2,"mode"],xlab="mean",ylab="mode",main="SC - Endox")
abline(b=1,a=0)
dev.off()

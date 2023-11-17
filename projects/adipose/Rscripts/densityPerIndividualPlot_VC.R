
raw_vc<-read.csv("/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/polygonsFiltered_forHistPerIndividualPlot/polygonsFiltered_vc_from1_to1000.txt",as.is=T,h=F,fill=T,col.names=paste0("V",seq_len(53184)))

colnames(raw_vc)[1:4]<-c("cohort","filename","N","below750")

raw_vc2<-raw_vc[ !is.na(raw_vc$V5),]


leip<-raw_vc2[ grepl("\\/a[0-9]",raw_vc2$filename),]
munich<-raw_vc2[ grepl("\\/m[0-9]",raw_vc2$filename),]
gtex<-raw_vc2[ grepl("\\/GTEX",raw_vc2$filename),]
hohen<-raw_vc2[ grepl("\\/h[0-9]",raw_vc2$filename),]

############################

png("densityPlotPerWSI_hohenheim.png",res=200)

ecdf1 <- density(na.omit(unlist(hohen[1,5:ncol(hohen)])))
# Plot all ECDFs on same plot
plot(ecdf1, main="ECDF Plot",  lty="dotted", lwd=2,col=rgb(red = 0, green = 0, blue = 0, alpha = 0.5),xlim=c(0,15000),ylim=c(0,0.002))

for(i in 2:nrow(hohen)){
      ecdf <- density(na.omit(unlist(hohen[i,5:ncol(hohen)])))
      lines(ecdf, lty="dotted", lwd=2,col=rgb(red = 0, green = 0, blue = 0, alpha = 0.5))
}

dev.off()

##############################

png("densityPlotPerWSI_munich.png",res=200)

ecdf1 <- density(na.omit(unlist(munich[1,5:ncol(munich)])))
# Plot all ECDFs on same plot
plot(ecdf1, main="ECDF Plot",  lty="dotted", lwd=2,col=rgb(red = 0, green = 0, blue = 0, alpha = 0.5),xlim=c(0,15000),ylim=c(0,0.002))

for(i in 2:nrow(munich)){
      ecdf <- density(na.omit(unlist(munich[i,5:ncol(munich)])))
      lines(ecdf, lty="dotted", lwd=2,col=rgb(red = 0, green = 0, blue = 0, alpha = 0.5))
}

dev.off()

##############################

png("densityPlotPerWSI_gtex.png",res=200)

ecdf1 <- density(na.omit(unlist(gtex[1,5:ncol(gtex)])))
# Plot all ECDFs on same plot
plot(ecdf1, main="ECDF Plot",  lty="dotted", lwd=2,col=rgb(red = 0, green = 0, blue = 0, alpha = 0.5),xlim=c(0,15000),ylim=c(0,0.002))

for(i in 2:nrow(gtex)){
      ecdf <- density(na.omit(unlist(gtex[i,5:ncol(gtex)])))
      lines(ecdf, lty="dotted", lwd=2,col=rgb(red = 0, green = 0, blue = 0, alpha = 0.5))
}

dev.off()

##############################

png("densityPlotPerWSI_leipzig.png",res=200)

ecdf1 <- density(na.omit(unlist(leip[1,5:ncol(leip)])))
# Plot all ECDFs on same plot
plot(ecdf1, main="ECDF Plot",  lty="dotted", lwd=2,col=rgb(red = 0, green = 0, blue = 0, alpha = 0.5),xlim=c(0,15000),ylim=c(0,0.002))

for(i in 2:nrow(leip)){
      ecdf <- density(na.omit(unlist(leip[i,5:ncol(leip)])))
      lines(ecdf, lty="dotted", lwd=2,col=rgb(red = 0, green = 0, blue = 0, alpha = 0.5))
}

dev.off()

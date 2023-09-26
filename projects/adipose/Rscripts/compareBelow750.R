
old750_sc<-read.csv("polygonsFiltered_forHistPerIndividualPlot/polygonsFiltered_sc_from1_to1000_first4cols.txt",as.is=T,h=F)
old750_vc<-read.csv("polygonsFiltered_forHistPerIndividualPlot/polygonsFiltered_vc_from1_to1000_first4cols.txt",as.is=T,h=F)

new750_sc<-read.csv("polygonsFiltered_sc_from1_to950V3_first4cols.txt",as.is=T,h=F)
new750_vc<-read.csv("polygonsFiltered_vc_from1_to950V3_first4cols.txt",as.is=T,h=F)


manual<-read.csv("manualGoodBadWSIV2.csv",as.is=T,h=T)

manual$Filename<-sapply(manual$File,function(x) tail(unlist(strsplit(x,"\\\\")),1))


endoxID <- read.csv("/gpfs3/well/lindgren/craig/isbi-2012/NDOG_histology_IDs_and_quality.csv",as.is=T)
endoxID_sc <- endoxID[ endoxID$depot %in% "subcutaneous",]

endox_sc <- sapply(basename(old750_sc$V2), function(x) x%in%endoxID_sc[,"filename"])
endox_vc <- sapply(basename(old750_sc$V2), function(x) !x%in%endoxID_sc[,"filename"] & grepl("Image",x))

old750_sc<-old750_sc[ grepl("[0-9]sc[0-9]",old750_sc$V2) | grepl("Subcutaneous",old750_sc$V2) | endox_sc,]
old750_vc<-old750_vc[ grepl("[0-9]vc[0-9]",old750_vc$V2) | grepl("Visceral",old750_vc$V2) | endox_vc,]



new750_sc<-new750_sc[ grepl("[0-9]sc[0-9]",new750_sc$V2) | grepl("Subcutaneous",new750_sc$V2) | endox_sc,]
new750_vc<-new750_vc[ grepl("[0-9]vc[0-9]",new750_vc$V2) | grepl("Visceral",new750_vc$V2) | endox_vc,]

manual_endox_sc <- sapply(manual$Filename, function(x) x%in%endoxID_sc[,"filename"])
manual_endox_vc <- sapply(manual$Filename, function(x) !x%in%endoxID_sc[,"filename"] & grepl("Image",x))

manual_sc<-manual[ grepl("[0-9]sc[0-9]",manual$Filename) | grepl("Visceral",manual$Filename) | manual_endox_sc,]
manual_vc<-manual[ grepl("[0-9]vc[0-9]",manual$Filename) | grepl("Visceral",manual$Filename) | manual_endox_vc,]


table(old750_sc$V2==new750_sc$V2)
table(old750_vc$V2==new750_vc$V2)




par(mfcol=c(1,2))
hist(old750_sc$V4,breaks=100,main="old > 0 - SC",xlab="old frac > 750")
abline(v=0.9)
hist(new750_sc$V4,breaks=100,main="new > 0 - SC",xlab="new frac > 750")
abline(v=0.65)
dev.off()

par(mfcol=c(1,2))
hist(old750_vc$V4,breaks=100,main="old > 0 - VC",xlab="new frac > 750")
abline(v=0.9)
hist(new750_vc$V4,breaks=100,main="new > 0 - VC",xlab="new frac > 750")
abline(v=0.65)
dev.off()


c_sc<-cbind(old=old750_sc[ old750_sc$V4>0.9,"V4"],new=new750_sc[ new750_sc$V2%in%old750_sc[ old750_sc$V4>0.9,"V2"],"V4"])
c_vc<-cbind(old=old750_vc[ old750_vc$V4>0.9,"V4"],new=new750_vc[ new750_vc$V2%in%old750_vc[ old750_vc$V4>0.9,"V2"],"V4"])

par(mfcol=c(1,2))
plot(c_sc[,1],c_sc[,2],xlim=c(0,1),ylim=c(0,1),xlab="old frac > 750",ylab="new frac > 750")
abline(h=0.65)
abline(v=0.9)
plot(c_vc[,1],c_vc[,2],xlim=c(0,1),ylim=c(0,1),xlab="old frac > 750",ylab="new frac > 750")
abline(h=0.65)
abline(v=0.9)
dev.off()



png("compareBelow750_smootherScatter_vc.png",res=200,units="in",width=12,height=7)
par(mfcol=c(1,2))
smoothScatter(old750_vc$V3,old750_vc$V4,ylab="frac < 750",xlab="N(after filter)",main="old - VC")
points(old750_vc[  basename(old750_vc$V2)%in%manual_vc[ manual_vc$Acceptable.quality%in%1,"Filename"],3:4],pch=4,col="blue",lwd=2)
points(old750_vc[  basename(old750_vc$V2)%in%manual_vc[ manual_vc$Low.quality%in%1,"Filename"],3:4],pch=4,col="red",lwd=2)
points(old750_vc[  basename(old750_vc$V2)%in%manual_vc[ manual_vc$Blurred%in%1,"Filename"],3:4],pch=4,col="violet",lwd=2)
points(old750_vc[  basename(old750_vc$V2)%in%manual_vc[ manual_vc$Empty%in%1,"Filename"],3:4],pch=4,col="black",lwd=2)
abline(h=0.9) 
legend("topright",c("low quality WSI","acceptable quality WSI","blurred WSI","empty WSI"),fill=c("red","blue","violet","black"))
smoothScatter(new750_vc$V3,new750_vc$V4,ylab="frac < 750",xlab="N(after filter)",main="new - VC")
points(new750_vc[  basename(new750_vc$V2)%in%manual_vc[ manual_vc$Acceptable.quality%in%1,"Filename"],3:4],pch=4,col="blue",lwd=2)
points(new750_vc[  basename(new750_vc$V2)%in%manual_vc[ manual_vc$Low.quality%in%1,"Filename"],3:4],pch=4,col="red",lwd=2)
points(new750_vc[  basename(new750_vc$V2)%in%manual_vc[ manual_vc$Blurred%in%1,"Filename"],3:4],pch=4,col="violet",lwd=2)
points(new750_vc[  basename(new750_vc$V2)%in%manual_vc[ manual_vc$Empty%in%1,"Filename"],3:4],pch=4,col="black",lwd=2)
abline(h=0.65)
abline(v=1500)
legend("topright",c("low quality WSI","acceptable quality WSI","blurred WSI","empty WSI"),fill=c("red","blue","violet","black"))
dev.off()

png("compareBelow750_smootherScatter_sc.png",res=200,units="in",width=12,height=7)
par(mfcol=c(1,2))
smoothScatter(old750_sc$V3,old750_sc$V4,ylab="frac < 750",xlab="N(after filter)",main="old - SC")
points(old750_sc[  basename(old750_sc$V2)%in%manual_sc[ manual_sc$Acceptable.quality%in%1,"Filename"],3:4],pch=4,col="blue",lwd=2)
points(old750_sc[  basename(old750_sc$V2)%in%manual_sc[ manual_sc$Low.quality%in%1,"Filename"],3:4],pch=4,col="red",lwd=2)
points(old750_sc[  basename(old750_sc$V2)%in%manual_sc[ manual_sc$Blurred%in%1,"Filename"],3:4],pch=4,col="violet",lwd=2)
points(old750_sc[  basename(old750_sc$V2)%in%manual_sc[ manual_sc$Empty%in%1,"Filename"],3:4],pch=4,col="black",lwd=2)
abline(h=0.9)
legend("topright",c("low quality WSI","acceptable quality WSI","blurred WSI","empty WSI"),fill=c("red","blue","violet","black"))
smoothScatter(new750_sc$V3,new750_sc$V4,ylab="frac < 750",xlab="N(after filter)",main="new - SC")
points(new750_sc[  basename(new750_sc$V2)%in%manual_sc[ manual_sc$Acceptable.quality%in%1,"Filename"],3:4],pch=4,col="blue",lwd=2)
points(new750_sc[  basename(new750_sc$V2)%in%manual_sc[ manual_sc$Low.quality%in%1,"Filename"],3:4],pch=4,col="red",lwd=2)
points(new750_sc[  basename(new750_sc$V2)%in%manual_sc[ manual_sc$Blurred%in%1,"Filename"],3:4],pch=4,col="violet",lwd=2)
points(new750_sc[  basename(new750_sc$V2)%in%manual_sc[ manual_sc$Empty%in%1,"Filename"],3:4],pch=4,col="black",lwd=2)
abline(h=0.65)
abline(v=1500)
legend("topright",c("low quality WSI","acceptable quality WSI","blurred WSI","empty WSI"),fill=c("red","blue","violet","black"))
dev.off()

print('manual_vc[ manual_vc$Acceptable.quality%in%0,]')
print(manual_vc[ manual_vc$Acceptable.quality%in%0,])
print(new750_vc[  basename(new750_vc$V2)%in%manual_vc[ manual_vc$Acceptable.quality%in%0,"Filename"],])
print("")
print('manual_sc[ manual_sc$Acceptable.quality%in%0,]')
print(manual_sc[ manual_sc$Acceptable.quality%in%0,])
print(new750_sc[  basename(new750_sc$V2)%in%manual_sc[ manual_sc$Acceptable.quality%in%0,"Filename"],])


print(dim(new750_vc[  new750_vc$V3<1500 &  new750_vc$V4>0.65,])/dim(new750_vc))
print(dim(new750_sc[  new750_sc$V3<1500 &  new750_sc$V4>0.65,])/dim(new750_sc))


print(new750_vc[  new750_vc$V3<1500 &  new750_vc$V4>0.65,])
print(new750_sc[  new750_sc$V3<1500 &  new750_sc$V4>0.65,])



write.table(new750_vc[  new750_vc$V3<3000 & new750_vc$V4>0.6,2],"wsisTopLeftCorner_vc_Nbelow3000_fracAbove060.list",col.names=F,row.names=F,qu=F)
write.table(new750_sc[  new750_sc$V3<3000 & new750_sc$V4>0.5,2],"wsisTopLeftCorner_sc_Nbelow3000_fracAbove060.list",col.names=F,row.names=F,qu=F)
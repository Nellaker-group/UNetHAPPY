



csv<-read.csv("multiRun_Total_from1_to927V2.csv",as.is=T)
head(csv[ grepl("m[0-9]",csv$filename) &  csv$avg<400,])
dim(csv[ grepl("m[0-9]",csv$filename) &  csv$avg<400,])






par(mfrow=c(1,2))
smoothScatter(csv$fracBelow750,csv$avg)
abline(v=400)
abline(v=300)
abline(v=200)
abline(v=6000)
abline(v=5000)
abline(v=10000)


smoothScatter(csv$fracBelow750[ csv$Nadipocytes < 5000],csv$avg[ csv$Nadipocytes < 5000])
abline(v=400)
abline(v=600)
abline(v=6000)
abline(v=5000)
abline(v=4000)


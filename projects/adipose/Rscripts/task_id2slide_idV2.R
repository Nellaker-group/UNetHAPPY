set.seed(21313)


c<-cbind(rep(1:826,each=10)[1:8259],1:8259)
c2 <- cbind(rep(1:826,each=10)[1:8259],sample(1:8259))
write.table(c2,"task_id2slide_id_31mar_shuffledV2.csv",col.names=F,qu=F,row.names=F,sep=",")

set.seed(21313)


## from finding the ones omitted from the db/main_04apr.db run
## /gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/findMissingWSIs.finalRun.py


missingPrevRun<-c(243, 1315, 1353, 1357, 1366, 1367, 1369, 1384, 1409, 1410, 1416, 1425, 1434, 1435, 1436, 1440, 1463, 1466, 1470, 1473, 1489, 1493, 1500, 1509, 1519, 1541, 1551, 1555, 1569, 1571, 1577, 1578, 1585, 1589, 1599, 1601, 1618, 1635, 1636, 1641, 1668, 1674, 1692, 1694, 1696, 1700, 1703, 1710, 1731, 1732, 1749, 1763, 1770, 1771, 1776, 1788, 1804, 1814, 1832, 1833, 1836, 1878, 1879, 1889, 1893, 1946, 1971, 2086, 2132, 2441, 2474, 2860, 3033, 3730, 3757, 3908, 4090, 4365, 4698, 4737, 4770, 5106, 5332, 6031, 6560, 6679, 7192, 7733, 7794)

newlyAddedSlides<-8260:9179

missingAndAddedSlides <- c(missingPrevRun,newlyAddedSlides)

c2 <- cbind(rep(827:927,each=10)[1:length(missingAndAddedSlides)],sample(missingAndAddedSlides))
write.table(c2,"task_id2slide_id_17apr_shuffledV2restOfSlides.csv",col.names=F,qu=F,row.names=F,sep=",")

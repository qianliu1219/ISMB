rm(list = ls())
setwd("~/Desktop/projects/Organoid/ISMB/code/cell_counting_v2-master")

############ performance on testing ###############



DRDCNN<-t(read.csv("DRDCNN_predict.csv",header = F))
DRDCNN[,2]<-unlist(strsplit(DRDCNN[,2],"]"))
DRDCNN[,2]<-substr(DRDCNN[,2],2,10)
DRDCNN<-as.data.frame(DRDCNN)
colnames(DRDCNN)<-c("label","DRDCNN")
DRDCNN[,1]<-as.numeric(as.character(DRDCNN$label))
DRDCNN[,2]<-as.numeric(as.character(DRDCNN$DRDCNN))



FRDCNN<-t(read.csv("FRDCNN_predict.csv",header = F))
FRDCNN[,2]<-unlist(strsplit(FRDCNN[,2],"]"))
FRDCNN[,2]<-substr(FRDCNN[,2],2,10)
FRDCNN<-as.data.frame(FRDCNN)
colnames(FRDCNN)<-c("label","FRDCNN")
FRDCNN[,1]<-as.numeric(as.character(FRDCNN$label))
FRDCNN[,2]<-as.numeric(as.character(FRDCNN$FRDCNN))



CRDCNN<-t(read.csv("CRDCNN_predict.csv",header = F))
CRDCNN[,2]<-unlist(strsplit(CRDCNN[,2],"]"))
CRDCNN[,2]<-substr(CRDCNN[,2],2,10)
CRDCNN<-as.data.frame(CRDCNN)
colnames(CRDCNN)<-c("label","ERDCNN")
CRDCNN[,1]<-as.numeric(as.character(CRDCNN$label))
CRDCNN[,2]<-as.numeric(as.character(CRDCNN$ERDCNN))



library(Metrics)

rmse(as.numeric(DRDCNN[,2]), as.numeric(DRDCNN[,1]))
mae(as.numeric(DRDCNN[,2]), as.numeric(DRDCNN[,1]))
cor(as.numeric(DRDCNN[,2]), as.numeric(DRDCNN[,1]),method = "pearson")

rmse(as.numeric(FRDCNN[,2]), as.numeric(FRDCNN[,1]))
mae(as.numeric(FRDCNN[,2]), as.numeric(FRDCNN[,1]))
cor(as.numeric(FRDCNN[,2]), as.numeric(FRDCNN[,1]),method = "pearson")

rmse(as.numeric(CRDCNN[,2]), as.numeric(CRDCNN[,1]))
mae(as.numeric(CRDCNN[,2]), as.numeric(CRDCNN[,1]))
cor(as.numeric(CRDCNN[,2]), as.numeric(CRDCNN[,1]),method = "pearson")




library(ggplot2)

p <-ggplot()+
  geom_point(data=DRDCNN, aes(x=DRDCNN, y=label,colour="DRDCNN"))  +
  geom_point(data=FRDCNN,aes(x=FRDCNN, y=label, colour="FRDCNN"))  +
  geom_point(data=CRDCNN,aes(x=ERDCNN, y=label, colour="ERDCNN"))

p + 
  xlab("Predicted cell counts") + 
  ylab("True cell counts") + 
  geom_abline() + lims(x = c(0,200), y = c(0,200))
ggsave("predict_true.png",width = 5, height = 3.5)


# density histogram
xt <- as.data.frame(table(myres_trans$label))
barplot(xt[,2] ,col="gray", xlab="cell counts",names.arg = xt[,1],space = 0.2, width = 1,
        main="Histogram with Normal Curve (Synthetic_set1)",xlim = c(1,200))


# real data
AKTP_CRDCNN<-read.csv("AKTP_CRDCNN_predict.csv",header = F)
AKTP_FRDCNN<-read.csv("AKTP_FRDCNN_predict.csv",header = F)
AKTP_DRDCNN<-read.csv("AKTP_DRDCNN_predict.csv",header = F)

AKTP<-data.frame(AKTP_DRDCNN,AKTP_FRDCNN,AKTP_CRDCNN)
colnames(AKTP)<-c("DRDCNN",'FRDCNN','ERDCNN')


AKTP_P2rx7_CRDCNN<-read.csv("AKTP_P2rx7_CRDCNN_predict.csv",header = F)
AKTP_P2rx7_FRDCNN<-read.csv("AKTP_P2rx7_FRDCNN_predict.csv",header = F)
AKTP_P2rx7_DRDCNN<-read.csv("AKTP_P2rx7_DRDCNN_predict.csv",header = F)

AKTP_P2rx7<-data.frame(AKTP_P2rx7_DRDCNN,AKTP_P2rx7_FRDCNN,AKTP_P2rx7_CRDCNN)
colnames(AKTP_P2rx7)<-c("DRDCNN",'FRDCNN','ERDCNN')

AKTP_Nt5e_CRDCNN<-read.csv("AKTP_Nt5e_CRDCNN_predict.csv",header = F)
AKTP_Nt5e_FRDCNN<-read.csv("AKTP_Nt5e_FRDCNN_predict.csv",header = F)
AKTP_Nt5e_DRDCNN<-read.csv("AKTP_Nt5e_DRDCNN_predict.csv",header = F)

AKTP_Nt5e<-data.frame(AKTP_Nt5e_DRDCNN,AKTP_Nt5e_FRDCNN,AKTP_Nt5e_CRDCNN)
colnames(AKTP_Nt5e)<-c("DRDCNN",'FRDCNN','ERDCNN')




################################################
################      t test    ################
################################################

t.test(AKTP$DRDCNN,AKTP_P2rx7$DRDCNN,var.equal=T)
t.test(AKTP$DRDCNN,AKTP_Nt5e$DRDCNN,var.equal=T)

t.test(AKTP$FRDCNN,AKTP_P2rx7$FRDCNN,var.equal=T)
t.test(AKTP$FRDCNN,AKTP_Nt5e$FRDCNN,var.equal=T)

t.test(AKTP$ERDCNN,AKTP_P2rx7$ERDCNN,var.equal=T)
t.test(AKTP$ERDCNN,AKTP_Nt5e$ERDCNN,var.equal=T)
################################################
###################    plot     ################
################################################

AKTP_DRDCNN<-cbind(AKTP$DRDCNN,rep("DRDCNN",length(AKTP$DRDCNN)))
AKTP_FRDCNN<-cbind(AKTP$FRDCNN,rep("FRDCNN",length(AKTP$FRDCNN)))
AKTP_CRDCNN<-cbind(AKTP$ERDCNN,rep("ERDCNN",length(AKTP$ERDCNN)))


AKTP_Plot<-as.data.frame(rbind(AKTP_DRDCNN,AKTP_FRDCNN,AKTP_CRDCNN))
AKTP_Plot<-as.data.frame(cbind(AKTP_Plot,rep("AKTP",nrow(AKTP_Plot))))
colnames(AKTP_Plot)<-c("counts","model","group")
AKTP_Plot$counts<-as.numeric(AKTP_Plot$counts)

AKTP_P2rx7_DRDCNN<-cbind(AKTP_P2rx7$DRDCNN,rep("DRDCNN",length(AKTP_P2rx7$DRDCNN)))
AKTP_P2rx7_FRDCNN<-cbind(AKTP_P2rx7$FRDCNN,rep("FRDCNN",length(AKTP_P2rx7$FRDCNN)))
AKTP_P2rx7_CRDCNN<-cbind(AKTP_P2rx7$ERDCNN,rep("ERDCNN",length(AKTP_P2rx7$ERDCNN)))
AKTP_P2rx7_Plot<-as.data.frame(rbind(AKTP_P2rx7_DRDCNN,AKTP_P2rx7_FRDCNN,AKTP_P2rx7_CRDCNN))
AKTP_P2rx7_Plot<-as.data.frame(cbind(AKTP_P2rx7_Plot,rep("AKTP_P2rx7",nrow(AKTP_P2rx7_Plot))))
colnames(AKTP_P2rx7_Plot)<-c("counts","model","group")
AKTP_P2rx7_Plot$counts<-as.numeric(AKTP_P2rx7_Plot$counts)


AKTP_Nt5e_DRDCNN<-cbind(AKTP_Nt5e$DRDCNN,rep("DRDCNN",length(AKTP_Nt5e$DRDCNN)))
AKTP_Nt5e_FRDCNN<-cbind(AKTP_Nt5e$FRDCNN,rep("FRDCNN",length(AKTP_Nt5e$FRDCNN)))
AKTP_Nt5e_CRDCNN<-cbind(AKTP_Nt5e$ERDCNN,rep("ERDCNN",length(AKTP_Nt5e$ERDCNN)))
AKTP_Nt5e_Plot<-as.data.frame(rbind(AKTP_Nt5e_DRDCNN,AKTP_Nt5e_FRDCNN,AKTP_Nt5e_CRDCNN))
AKTP_Nt5e_Plot<-as.data.frame(cbind(AKTP_Nt5e_Plot,rep("AKTP_Nt5e",nrow(AKTP_Nt5e_Plot))))
colnames(AKTP_Nt5e_Plot)<-c("counts","model","group")
AKTP_Nt5e_Plot$counts<-as.numeric(AKTP_Nt5e_Plot$counts)


plot<-as.data.frame(rbind(AKTP_Plot,AKTP_P2rx7_Plot,AKTP_Nt5e_Plot))

plot_DRDCNN<-plot[plot$model=="DRDCNN",]
plot_FRDCNN<-plot[plot$model=="FRDCNN",]
plot_CRDCNN<-plot[plot$model=="ERDCNN",]

plot_AKTP<-plot[plot$group=="AKTP",]
plot_AKTP_P2rx7<-plot[plot$group=="AKTP_P2rx7",]
plot_AKTP_Nt5e<-plot[plot$group=="AKTP_Nt5e",]

p  <- ggplot(plot_DRDCNN, aes(counts, colour=group, fill=group))
p  <- p + geom_density(alpha=0.55)+ ggtitle("DRDCNN")+scale_color_hue(l=40, c=35)+ scale_fill_hue(l=40, c=35)
ggsave("DRDCNN.png",width = 5, height = 3.5)

p  <- ggplot(plot_FRDCNN, aes(counts, colour=group, fill=group))
p  <- p + geom_density(alpha=0.55)+ ggtitle("FRDCNN")+scale_color_hue(l=40, c=35)+ scale_fill_hue(l=40, c=35)
ggsave("FRDCNN.png",width = 5, height = 3.5)

p  <- ggplot(plot_CRDCNN, aes(counts, colour=group, fill=group))
p  <- p + geom_density(alpha=0.55)+ ggtitle("ERDCNN")+scale_color_hue(l=40, c=35)+ scale_fill_hue(l=40, c=35)
ggsave("ERDCNN.png",width = 5, height = 3.5)

p  <- ggplot(plot_AKTP, aes(counts, colour=model, fill=model))
p  <- p + geom_density(alpha=0.55)+ ggtitle("AKTP")
ggsave("AKTP.png",width = 5, height = 3.5)

p  <- ggplot(plot_AKTP_P2rx7, aes(counts, colour=model, fill=model))
p  <- p + geom_density(alpha=0.55)+ ggtitle("AKTP_P2rx7")
ggsave("AKTP_P2rx7.png",width = 5, height = 3.5)

p  <- ggplot(plot_AKTP_Nt5e, aes(counts, colour=model, fill=model))
p  <- p + geom_density(alpha=0.55)+ ggtitle("AKTP_Nt5e")
ggsave("AKTP_Nt5e.png",width = 5, height = 3.5)






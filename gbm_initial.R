# Initial gradient boosting model for Higgs Boson
require(doMC)
registerDoMC(cores=7)
library(data.table)
library(gbm)

# Read the training data
input <- fread("training.csv")

# Check if the class of the input data is correct
lapply(input,class)
# Set the target as factor variable
input$Label <- as.factor(input$Label)

# Read the test data
test <- fread("test.csv")
test <- data.frame(test)

input <- data.frame(input)
train <- input[,c(2:31,33)]

# Replace all -999 with NA. Then impute the columns with their mean

#comb_row[,10:749] <- apply(comb_row[,10:749], 2, function(x){x <- replace(x, is.na(x), 0)}) 
train[,1:30] <- apply(train[,1:30],2,function(x){x <- replace(x,x==-999,NA)})
train[,1:30] <- apply(train[,1:30],2,function(x){x <- replace(x,is.na(x),mean(x,na.rm=T))})
train$label <- as.numeric(train$Label)
train$label <- train$label - 1
train <- train[,-31]

# Fit a Gradient Boosting Model

system.time(model_gbm_1 <- gbm(
	label~.,
	data=train,
	distribution="bernoulli",
	n.trees=500,
	interaction.depth=10,
	cv.folds=7,
	shrinkage=0.001,
	bag.fraction=0.7,
	train.fraction=0.8,
	verbose="CV",
	class.stratify.cv=T,
	n.cores=7
	))

# Takes 1220 seconds

# Apply the same imputation to Test dataset
test[,2:31] <- apply(test[,2:31],2,function(x){x <- replace(x,x==-999,NA)})
test[,2:31] <- apply(test[,2:31],2,function(x){x <- replace(x,is.na(x),mean(x,na.rm=T))})

# Score the test dataset on the model
output_gbm_1 <- predict(model_gbm_1,test[,-1],type="response")

output_gbm_1 <- data.frame(output_gbm_1)
# Based on prob, set s or b
output_gbm_1$label <-  ifelse(output_gbm_1[,1]>0.5,"s","b")

# Write output
write.csv(output_gbm_1,"gbm_output_29jul.csv",row.names=FALSE)


# Leaderboard score: Around 2.7

# From GBM:The following columns were significant

#DER_mass_transverse_met_lep DER_mass_transverse_met_lep 37.694441860
#DER_mass_MMC                               DER_mass_MMC 36.922466833
#DER_mass_vis                               DER_mass_vis 10.620958577
#DER_met_phi_centrality           DER_met_phi_centrality  5.904412198
#PRI_tau_pt                                   PRI_tau_pt  5.064079914
#DER_deltar_tau_lep                   DER_deltar_tau_lep  1.988303382
#DER_deltaeta_jet_jet               DER_deltaeta_jet_jet  1.700366604
#DER_lep_eta_centrality           DER_lep_eta_centrality  0.055271006
#DER_prodeta_jet_jet                 DER_prodeta_jet_jet  0.024034053
#DER_pt_h                                       DER_pt_h  0.011925557
#DER_mass_jet_jet                       DER_mass_jet_jet  0.008039462
#DER_sum_pt                                   DER_sum_pt  0.005700554

# Results
library(dplyr)
library(ggplot2)
library(lubridate)
library(ROCR)
library(knitr)
library(rpart)
library(rpart.plot)
library(pander)
library(ROCit)
library(grid)
library(gridExtra)
library(grDevices)
library(fpc)
library(ggpubr)
library(factoextra)
youtube <- read.csv('Global YouTube Statistics.csv')
#there are 56 NAs
summary(youtube$video_views_for_the_last_30_days)     #there are 56 NAs
youtube <-filter(youtube, !is.na(video_views_for_the_last_30_days))   #excluding NAs

# replace NA with "other" for category and channel_type columns
youtube$category[is.na(youtube$category)] <- "Other"
youtube$channel_type[is.na(youtube$channel_type)] <- "Other"

# replace nan with other" category and channel_type columns
youtube$category[youtube$category == "nan"] <- "Other"
youtube$channel_type[youtube$channel_type == "nan"] <- "Other"

# calculate the median of subscribers_for_last_30_days
median_value <- median(youtube$subscribers_for_last_30_days, na.rm = TRUE)

# replace NaN with the median
youtube$subscribers_for_last_30_days[youtube$subscribers_for_last_30_days == "NaN"] <- median_value

replaceTopBottomOutliers <- function(data, column, lower_percentile, upper_percentile) {
  # Calculate the quantiles for the specified column
  thresholds <- quantile(data[[column]], probs = c(lower_percentile, upper_percentile), na.rm = TRUE)
  
  # Replace values in the specified column that are above the upper threshold
  data[[column]][data[[column]] > thresholds[2]] <- median(data[[column]], na.rm = TRUE)
  
  # Replace values in the specified column that are below the lower threshold
  data[[column]][data[[column]] < thresholds[1]] <- median(data[[column]], na.rm = TRUE)
  return(data)
}

# Example usage:
youtube <- replaceTopBottomOutliers(youtube, "highest_monthly_earnings", 0.25, 0.8)
youtube <- replaceTopBottomOutliers(youtube, "lowest_monthly_earnings", 0.2, 0.8)
youtube <- replaceTopBottomOutliers(youtube, "highest_yearly_earnings", 0.25, 0.8)
youtube <- replaceTopBottomOutliers(youtube, "lowest_yearly_earnings", 0.2, 0.75)
youtube <- replaceTopBottomOutliers(youtube, "uploads", 0.1, 1)
youtube <- replaceTopBottomOutliers(youtube, "subscribers", 0, 1)
youtube <- replaceTopBottomOutliers(youtube, "video.views", 0.04, 1)
youtube <- replaceTopBottomOutliers(youtube, "subscribers_for_last_30_days", 0.2, 0.9)
youtube <- replaceTopBottomOutliers(youtube, "created_date", 0.1, 0.9)

youtube$vv <- ifelse(youtube$video_views_for_the_last_30_days > 80000000, 1, 0)   #1 for hige video views, 2 for low video views
table(youtube$vv)

youtube$created <- as.POSIXct("2023-10-20") 
year(youtube$created) <- as.integer(youtube$created_year)
youtube$created_month <- as.factor(youtube$created_month)
youtube$created_month <- factor(youtube$created_month, levels = c('Jan', 'Feb', 'Mar',
                                                                  'Apr', 'May', 'Jun',
                                                                  'Jul', 'Aug', 'Sep',
                                                                  'Oct', 'Nov', 'Dec'), 
                                labels = c('01', '02', '03', '04', '05', '06',
                                           '07', '08', '09', '10', '11', '12'))
month(youtube$created) <- as.integer(youtube$created_month)
mday(youtube$created) <- as.integer(youtube$created_date)       #combine year, month and day
youtube$created_time <- as.POSIXct("2023-10-20") - youtube$created      #calculate total opening days
youtube$created_time <- as.numeric(youtube$created_time) 
youtube <- rename(youtube, subscribers_30 = 'subscribers_for_last_30_days',
                  edu = 'Gross.tertiary.education.enrollment....')      #rename column names
candidate.columns <- c('subscribers', 'video.views', 'category', 'uploads', 
                       'channel_type', 'lowest_monthly_earnings',
                       'highest_monthly_earnings', 'lowest_yearly_earnings', 
                       'highest_yearly_earnings', 'subscribers_30', 'created_time')
predict1.columns <- c('subscribers', 'category', 'uploads', 'channel_type', 'created_time')
predict2.columns <- c('lowest_yearly_earnings', 'video.views', 'subscribers_30', 'uploads', 'category')
outcome <- 'vv'
youtube <- youtube[, c(outcome, candidate.columns)]
catVars <- candidate.columns[sapply(youtube[,candidate.columns],class) %in% c('factor','character')]
numericVars <- candidate.columns[sapply(youtube[,candidate.columns],class) %in% c('numeric','integer')]

set.seed(9924)
youtube$rgroup <- runif(dim(youtube)[1])
TrainAll <- subset(youtube, rgroup<=0.9)
Test <- subset(youtube, rgroup>0.9)
useForCal <- rbinom(n=dim(TrainAll)[1], size=1, prob=0.1)>0
Cal <- subset(TrainAll, useForCal)
Train <- subset(TrainAll, !useForCal)

#Single variable model performance
pos <- '1'
mkPredC <- function(outCol, varCol, appCol) {
  pPos <- sum(outCol == pos) / length(outCol)
  naTab <- table(as.factor(outCol[is.na(varCol)]))
  pPosWna <- (naTab/sum(naTab))[pos]
  vTab <- table(as.factor(outCol), varCol)
  pPosWv <- (vTab[pos, ] + 1.0e-3*pPos) / (colSums(vTab) + 1.0e-3)
  pred <- pPosWv[appCol]
  pred[is.na(appCol)] <- pPosWna
  pred[is.na(pred)] <- pPos
  pred
} 
for(v in catVars) {
  pi <- paste('pred', v, sep='')
  Train[,pi] <- mkPredC(Train[,outcome], Train[,v], Train[,v])
  Cal[,pi] <- mkPredC(Train[,outcome], Train[,v], Cal[,v])
  Test[,pi] <- mkPredC(Train[,outcome], Train[,v], Test[,v])
}
calcAUC <- function(predcol,outcol) {
  perf <- performance(prediction(predcol,outcol==pos),'auc')
  as.numeric(perf@y.values)
}
for(v in catVars) {
  pi <- paste('pred', v, sep='')
  aucTrain <- calcAUC(Train[,pi], Train[,outcome])
  if (aucTrain >= 0.6) {
    aucCal <- calcAUC(Cal[,pi], Cal[,outcome])
    print(sprintf(
      "%s: trainAUC: %4.3f; calibrationAUC: %4.3f",
      pi, aucTrain, aucCal))
  }
}
mkPredN <- function(outCol, varCol, appCol) {
  cuts <- unique(
    quantile(varCol, probs=seq(0, 1, 0.1), na.rm=T))
  varC <- cut(varCol,cuts)
  appC <- cut(appCol,cuts)
  mkPredC(outCol,varC,appC)
}
for(v in numericVars) {
  pi <- paste('pred', v, sep='')
  Train[,pi] <- mkPredN(Train[,outcome], Train[,v], Train[,v])
  Cal[,pi] <- mkPredN(Train[,outcome], Train[,v], Cal[,v])
  Test[,pi] <- mkPredN(Train[,outcome], Train[,v], Test[,v])
  aucTrain <- calcAUC(Train[,pi], Train[,outcome])
  if(aucTrain >= 0.5) {
    aucCal <- calcAUC(Cal[,pi], Cal[,outcome])
    print(sprintf(
      "%s: trainAUC: %4.3f; calibrationAUC: %4.3f",
      pi, aucTrain, aucCal))
  }
}
all.vars <- c(catVars, numericVars)
models.auc <- data.frame(model.type = 'univariate',
                         model.name = all.vars,
                         train.auc = sapply(all.vars, function(v){pi <- paste('pred',v,sep=''); 
                         calcAUC(Train[,pi], Train[,outcome])}),
                         cal.auc = sapply(all.vars, function(v){pi <- paste('pred',v,sep=''); 
                         calcAUC(Cal[,pi],Cal[,outcome])}))

#Two classification model performance
plot_roc <- function(predcol1, outcol1, predcol2, outcol2){
  roc_1 <- rocit(score=predcol1, class=outcol1==pos)
  roc_2 <- rocit(score=predcol2, class=outcol2==pos)
  plot(roc_1, col = c("blue","green"), lwd = 3,
       legend = FALSE,YIndex = FALSE, values = TRUE, asp=1)
  lines(roc_2$TPR ~ roc_2$FPR, lwd = 3,
        col = c("red","green"), asp=1)
  legend("bottomright", col = c("blue","red", "green"),
         c("Test Data", "Training Data", "Null Model"), lwd = 2)
}
#model results
fV1 <- paste(outcome,'>0 ~ ', paste(predict1.columns, collapse=' + '), sep='')
tmodel_play <- rpart(fV1, data=Train)
predict1.columns_log <- c('subscribers', 'video.views', 'uploads', 'created_time')
fL1 <- paste(outcome, paste(predict1.columns_log, collapse=" + "), sep=" ~ ")
gmodel_play <- glm(fL1, data=Train, family=binomial(link="logit"))
fV2 <- paste(outcome,'>0 ~ ', paste(predict2.columns, collapse=' + '), sep='')
tmodel_earning <- rpart(fV2, data=Train)
fL2 <- paste(outcome, paste(predict2.columns, collapse=" + "), sep=" ~ ")
gmodel_earning <- glm(fL2, data=Train, family=binomial(link="logit"))

pred_tplay <- prediction(predict(tmodel_play, newdata=Test), Test[[outcome]])
pred_gplay <- prediction(predict(gmodel_play, newdata=Test), Test[[outcome]])
pred_tearn <- prediction(predict(tmodel_earning, newdata=Test), Test[[outcome]])
pred_pearn <- prediction(predict(gmodel_earning, newdata=Test), Test[[outcome]])
preds <- list(pred_tplay, pred_gplay, pred_tearn, pred_pearn)
model.names <- c('decision tree_video play', 'logistic regression_video play', 
                 'decision tree_video earning', 'logistic regression_video earning')
roc.df <- data.frame()
for(i in 1:length(model.names)){
  pred <- preds[[i]]
  model.name <- model.names[i]
  perf <- performance( pred, "tpr", "fpr" )
  roc.df <- rbind(roc.df, data.frame(
    'fpr'=unlist(perf@x.values),
    'tpr'=unlist(perf@y.values),
    'threshold'=unlist(perf@alpha.values),
    'model'=model.name))
}

#Clustering results
youtube2 <- read.csv('Global YouTube Statistics.csv')
youtube2 <- youtube2[, c('category', 'subscribers', 'video.views', 'video_views_for_the_last_30_days',
                         'lowest_monthly_earnings', 'highest_monthly_earnings')]
youtube2 <-filter(youtube2, !is.na(video_views_for_the_last_30_days))
youtube2$category[youtube2$category == "nan"] <- "Other"
youtube2 <- replaceTopBottomOutliers(youtube2, "highest_monthly_earnings", 0.02, 1)
youtube2 <- replaceTopBottomOutliers(youtube2, "lowest_monthly_earnings", 0.15, 1)
youtube2 <- replaceTopBottomOutliers(youtube2, "subscribers", 0, 1)
youtube2 <- replaceTopBottomOutliers(youtube2, "video.views", 0.02, 1)
youtube2 <- replaceTopBottomOutliers(youtube2, "video_views_for_the_last_30_days", 0.05, 1)

vars.to.use <- colnames(youtube2)[-1]
scaled_df <- scale(youtube2[,vars.to.use])
d <- dist(scaled_df, method="euclidean")
pfit <- hclust(d, method="ward.D2")
groups <- cutree(pfit, k=7)
kmClusters2 <- kmeans(scaled_df, 9, nstart=100, iter.max=100)

# Define UI for application
ui <- fluidPage(
  # Application title
  titlePanel("Project 2 — Modelling"),
  
  # Sidebar layout with input and output definitions
  sidebarLayout(
    sidebarPanel(
      # Define the select input for selecting the distribution type
      selectInput("result", "Choose an option",
                  choices = list("Single variable model performance" = list("Single variable model performance" = "svmp"),
                                 "Two classification model performance" = list("Decision Tree - channel metrics as predictors" = "dtcm",
                                                                               "Logistic Regression - channel metrics as predictors" = "lrcm",
                                                                               "Decision Tree - earning estimates as predictors" = "dtee",
                                                                               "Logistic Regression - earning estimates as predictors" = "lree",
                                                                               "Comparision on four models" = "ct"),
                                 "Clustering results" = list("Hierarchical Clustering" = "hc",
                                                             "kMeans Clustering" = "kc")))
    ),
    
    # Show the plot based on selected distribution type
    mainPanel(
      tableOutput('table'),
      plotOutput('plot')
    )
  )
)

# Define server logic 
server <- function(input, output) {
  output$table <- renderTable({
    if (input$result == "svmp") {
      models.auc[order(-models.auc$cal.auc), ]
      } 
    })
  output$plot <- renderPlot({
    if (input$result == "dtcm") {
      plot_roc(predict(tmodel_play, newdata=Test), Test[[outcome]],
               predict(tmodel_play, newdata=Train), Train[[outcome]])
    } else if (input$result == "lrcm") {
      plot_roc(predict(gmodel_play, newdata=Test), Test[[outcome]],
               predict(gmodel_play, newdata=Train), Train[[outcome]])
    } else if (input$result == "dtee") {
      plot_roc(predict(tmodel_earning, newdata=Test), Test[[outcome]],
               predict(tmodel_earning, newdata=Train), Train[[outcome]])
    }else if (input$result == "lree") {
      plot_roc(predict(gmodel_earning, newdata=Test), Test[[outcome]],
               predict(gmodel_earning, newdata=Train), Train[[outcome]])
    } else if (input$result == "ct") {
      ggplot(roc.df, aes(x=fpr, y=tpr, group=model, color=model))+
        geom_line(size = 1) +
        geom_abline(intercept = 0, slope = 1, size = 1, 
                    color = 'green', linetype='dashed') +
        theme_bw()
    } else if (input$result == "hc") {
      fviz_cluster(list(data = scaled_df, cluster = groups),
                   geom = "point",
                   ellipse.type = "convex", 
                   ggtheme = theme_bw()
      )
    } else if (input$result == "kc") {
      fviz_cluster(kmClusters2, data = scaled_df,
                   geom = "point",
                   ellipse.type = "convex", 
                   ggtheme = theme_bw()
      )
    } 
  })
}

# Run the application 
shinyApp(ui = ui, server = server)

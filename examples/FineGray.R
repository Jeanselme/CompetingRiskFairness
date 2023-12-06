# This script runs Fine Gray on all datasets

library(riskRegression)
library(prodlim)
library(survival)
library(cmprsk)
library(readr)

# Open data
data = read_csv('tmp.csv')

# Create matrix results for both Fine Gray and Cox
prediction_fg = matrix(NA, nrow = nrow(data), ncol = 2 * 100 + 1) # outcomes * Times
rownames(prediction_fg) = as.numeric(rownames(data)) - 1
eval_times = seq(min(data['Time']), max(data['Time']), length.out = 100)
colnames(prediction_fg) = c(eval_times, eval_times, 'Use')
prediction_cs = prediction_fg # Make a copy
prediction_nc = prediction_fg

var_tot = setdiff(colnames(data), c("Time", "Event", "Fold_0"))
data_folder = subset(data, data["Fold_0"] == "Train")[append(var_tot, c('Time', 'Event'))]

# Create associated formula
var = setdiff(colnames(data_folder), c("Time", "Event"))
formula_nc = reformulate(var, response = "Surv(Time, Event)") 
formula = reformulate(var, response = "Hist(Time, Event)") 
print(formula)

# Fit model 
for (outcome in 1:2) {
    test = (data["Fold_0"] == "Test")

    # Run Fine Gray
    tryCatch(
        expr = {
                start_time <- Sys.time()
                model = FGR(formula, data = data_folder, cause = outcome)
                training_timefg <- Sys.time() - start_time
                # Predict at the time horizons of interest CSC + predictRisk
                prediction_fg[test,((outcome-1)*100+1):(outcome*100)] = 1 - predict(model, subset(data, test), eval_times, cause = outcome)
            },
        error = function(e){ 
            print(e)
            prediction_fg[test,((outcome-1)*100+1):(outcome*100)] = NA
        })
    prediction_fg[test, ncol(prediction_fg)] = 0.

    # Run Cause specific Cox
    tryCatch(
        expr = {
                start_time <- Sys.time()
                model = CSC(formula, data = data_folder, cause = outcome, method = "breslow", fitter = "cph", iter = 100)
                training_timecsc <- Sys.time() - start_time
                # Predict at the time horizons of interest CSC + predictRisk
                prediction_cs[test,((outcome-1)*100+1):(outcome*100)] = 1 - predictRisk(model, subset(data, test), eval_times, cause = outcome)
            },
        error = function(e){ 
            print(e)
            prediction_cs[test,((outcome-1)*100+1):(outcome*100)] = NA
        })
    prediction_cs[test, ncol(prediction_cs)] = 0.
    # Run Cox
    tryCatch(
        expr = {
                data_folder_nc = data_folder
                data_folder_nc$Event = (data_folder_nc$Event == outcome)
                start_time <- Sys.time()
                model = coxph(formula_nc, data = data_folder_nc, x = TRUE)
                training_timecox <- Sys.time() - start_time
                # Predict at the time horizons of interest CSC + predictRisk
                prediction_nc[test,((outcome-1)*100+1):(outcome*100)] = 1 - predictRisk(model, subset(data, test), eval_times)
            },
        error = function(e){ 
            print(e)
            prediction_nc[test,((outcome-1)*100+1):(outcome*100)] = NA
        })
    prediction_nc[test, ncol(prediction_nc)] = 0.
}
# Save
write.csv(prediction_fg, 'tmp_finegray.csv', na = '')
write.csv(prediction_cs, 'tmp_coxcs.csv', na = '')
write.csv(prediction_nc, 'tmp_cox.csv', na = '')

data <- data.frame(
  Model = c("Fine Gray", "CS Cox", "Cox"),
  running_time = c(as.numeric(training_timefg), as.numeric(training_timecsc), as.numeric(training_timecox))
)

write.csv(data, file = "tmp_times.csv", row.names = FALSE)

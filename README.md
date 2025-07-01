How It Works
	Training
•	Runs for 150 epochs.
•	Uses a grain yield forecasting model that combines AMD + TDI.
•	Agricultural Mode Decomposition (AMD)
•	Temporal Depthwise Informer (TDI)
•	Loss function: MSELoss(), optimized using Adam.
	Testing
•	After training finishes, the model is evaluated on the 15% test data.
•	Computes:
•	R2 (Coefficient of determination)
•	MAE (Mean absolute error)
•	MSE (Mean squared error)
•	RMSE(Root mean squared error)
•	MAPE(Mean absolute percentage error)
•	sMAPE(Symmetric mean absolute percentage error)
•	NSE(Nash sutcliffe efficiency)
	Model Saving
•	Saving the best model by crop
model_path = "best_model_{crop}.pth"
torch.save(model.state_dict(), model_path)
	Results Saving
pd.DataFrame(results).to_csv("src/csvfiles/output.csv", index=False)
	Dataset 
•	The China Statistical Yearbook data is collected from (https://www.stats.gov.cn/sj/ndsj/2024/indexeh.htm).
•	Load the dataset:
load_cropwise_normalized_data("grain_yield.csv")
•	Replace “grain_yield.csv” with actual dataset file path
How to Run Training 
train_model(model, train_loader, val_loader, optimizer, criterion, epochs, crop)
How to Run Testing
test_model(model, f" best_model_{crop}.pth", test_loader, crop, scaler_dict)
•	Replace ”best_model_{crop}.pth” with trained model.


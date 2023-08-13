// Import the necessary ML.NET namespace
using CustomerChurnMLDemo;
using Microsoft.ML;

// Create an MLContext instance, which serves as the entry point to ML.NET functionality
var mlContext = new MLContext();

// Load the data from a file using the CustomerChurn class's LoadIDataViewFromFile method
// It loads the data for retraining a machine learning model for customer churn prediction
var data = CustomerChurn.LoadIDataViewFromFile(mlContext, CustomerChurn.RetrainFilePath, CustomerChurn.RetrainSeparatorChar, CustomerChurn.RetrainHasHeader);

// Get the full path to the model file "CustomerChurn.mlnet"
string modelPath = Path.GetFullPath("CustomerChurn.mlnet");

// Load a pre-trained ML.NET model from the modelPath and get the ITransformer
// The model is loaded for further usage, like prediction or feature importance calculation
ITransformer model = mlContext.Model.Load(modelPath, out var _);

// Calculate Permutation Feature Importance (PFI) using the CustomerChurn class's CalculatePFI method
// PFI assesses the impact of each feature on the prediction results of the model
var pfi = CustomerChurn.CalculatePFI(mlContext, data, model, "Churn");

// Iterate through each tuple (feature, importance score) in the PFI results
foreach (var tuple in pfi)
{
    // Print the feature name and its importance score
    Console.WriteLine($"{tuple.Item1} - {tuple.Item2}");
}
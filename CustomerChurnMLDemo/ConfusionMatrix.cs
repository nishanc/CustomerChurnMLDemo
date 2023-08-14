using Microsoft.ML.AutoML;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace CustomerChurnMLDemo;

public static class ConfusionMatrix
{
    public static void CreateMatrix()
    {
        string trainDataFilePath = @"D:\CustomerChurnMLDemo\CustomerChurnMLDemo\Data\customer_churn_dataset-training-master.csv";
        string testDataFilePath = @"D:\CustomerChurnMLDemo\CustomerChurnMLDemo\Data\customer_churn_dataset-testing-master.csv";

        // Initialize a new MLContext
        var context = new MLContext();

        // Load your dataset
        IDataView dataView = context.Data.LoadFromTextFile<CustomerChurn.ModelInput>(trainDataFilePath, separatorChar: ',', hasHeader: true);

        // Load your evaluation/test data
        IDataView testDataView = context.Data.LoadFromTextFile<CustomerChurn.ModelInput>(testDataFilePath, separatorChar: ',', hasHeader: true);

        // Define the AutoML experiment settings
        MulticlassExperimentSettings settings = new MulticlassExperimentSettings()
        {
            OptimizingMetric = MulticlassClassificationMetric.MacroAccuracy,
            MaxExperimentTimeInSeconds = 600,
            CacheDirectoryName = null, // Skip the disk and store in-memory
        };

        // Run AutoML experiment
        var experiment = context.Auto().CreateMulticlassClassificationExperiment(settings);
        var result = experiment.Execute(dataView, validationData: testDataView, labelColumnName: "Churn", progressHandler: new MulticlassProgressReporter());

        // Get the best model
        var bestModel = result.BestRun.Model;

        // Make predictions using the best model
        var predictions = bestModel.Transform(testDataView);

        // Evaluate model's performance
        var metrics = context.MulticlassClassification.Evaluate(predictions, "Churn");

        // Print confusion matrix
        Console.WriteLine($"Confusion Matrix:\n{metrics.ConfusionMatrix.GetFormattedConfusionTable()}");

        // Print other performance metrics
        Console.WriteLine($"Accuracy: {metrics.MacroAccuracy}");
        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}");
        Console.WriteLine($"LogLoss: {metrics.LogLoss}");
    }
}

public class MulticlassProgressReporter : IProgress<RunDetail<MulticlassClassificationMetrics>>
{
    public void Report(RunDetail<MulticlassClassificationMetrics> value)
    {
        // Metrics may be null if an exception occurred or this run was canceled due to time constraints
        if (value.ValidationMetrics != null)
        {
            double accuracy = value.ValidationMetrics.MacroAccuracy;

            Console.WriteLine($"{value.TrainerName} ran in {value.RuntimeInSeconds:0.00} seconds with accuracy of {accuracy:p}");
        }
        else
        {
            Console.WriteLine($"{value.TrainerName} ran in {value.RuntimeInSeconds:0.00} seconds but did not complete. Time likely expired.");
        }
    }
}
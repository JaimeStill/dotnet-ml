using System;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace MulticlassClassification
{
    class Program
    {
        private static string appPath => Path.Combine(Environment.CurrentDirectory);
        private static string trainDataPath => Path.Combine(appPath, "Data", "issues_train.tsv");
        private static string testDataPath => Path.Combine(appPath, "Data", "issues_test.tsv");
        private static string modelPath => Path.Combine(appPath, "Models", "model.zip");

        private static MLContext mL;
        private static PredictionEngine<GitHubIssue, IssuePrediction> engine;
        private static ITransformer model;
        static IDataView trainingDataView;

        static void Main(string[] args)
        {
            mL = new MLContext(seed: 0);

            trainingDataView = mL.Data
                .LoadFromTextFile<GitHubIssue>(trainDataPath, hasHeader: true);

            var pipeline = ProcessData();
            var trainingPipeline = BuildAndTrainModel(trainingDataView, pipeline);

            Evaluate(trainingDataView.Schema);
            SaveModelAsFile(trainingDataView.Schema);
            PredictIssue();
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = mL.Transforms
                .Conversion
                .MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(mL.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(mL.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(mL.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(mL);

            return pipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(mL.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(mL.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            model = trainingPipeline.Fit(trainingDataView);
            engine = mL.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(model);

            var issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like it is going slow in my development machine."
            };

            var prediction = engine.Predict(issue);

            Console.WriteLine($"Single Prediction just-trained-model - Result: {prediction.Area}");

            return trainingPipeline;
        }

        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testDataView = mL.Data
                .LoadFromTextFile<GitHubIssue>(testDataPath, hasHeader: true);

            var metrics = mL.MulticlassClassification.Evaluate(model.Transform(testDataView));

            Console.WriteLine();
            Console.WriteLine("Metrics for Multi-class Classification model - Test Data");
            Console.WriteLine();
            Console.WriteLine($"MicroAccuracy:    {metrics.MicroAccuracy:0.###}");
            Console.WriteLine($"MacroAccuracy:    {metrics.MacroAccuracy:0.###}");
            Console.WriteLine($"LogLoss:          {metrics.LogLoss:#.###}");
            Console.WriteLine($"LogLossReduction: {metrics.LogLossReduction:#.###}");
            Console.WriteLine();
        }

        private static void PredictIssue()
        {
            ITransformer loadedModel = mL.Model.Load(modelPath, out var modelInputSchema);

            var issue = new GitHubIssue
            {
                Title = "Entity Framework crashes",
                Description = "When connecting to the database, EF is crashing"
            };

            engine = mL.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
            var prediction = engine.Predict(issue);

            Console.WriteLine($"Single Prediction - Result: {prediction.Area}");
        }

        private static void SaveModelAsFile(DataViewSchema trainingDataViewSchema)
        {
            mL.Model.Save(model, trainingDataViewSchema, modelPath);
        }
    }
}

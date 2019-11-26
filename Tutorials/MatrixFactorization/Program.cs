using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace MatrixFactorization
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mL = new MLContext();
            var views = LoadData(mL);
            var model = BuildAndTrainModel(mL, views.training);
            EvaluateModel(mL, views.test, model);
            UseModelForSinglePrediction(mL, model);
            SaveModel(mL, views.training.Schema, model);
        }

        public static (IDataView training, IDataView test) LoadData(MLContext mL)
        {
            var trainingPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
            var testPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

            IDataView trainingDataView = mL.Data.LoadFromTextFile<MovieRating>(trainingPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mL.Data.LoadFromTextFile<MovieRating>(testPath, hasHeader: true, separatorChar: ',');

            return (trainingDataView, testDataView);
        }

        public static ITransformer BuildAndTrainModel(MLContext mL, IDataView trainingDataView)
        {
            IEstimator<ITransformer> estimator = mL.Transforms
                .Conversion
                .MapValueToKey(outputColumnName: "UserIdEncoded", inputColumnName: "UserId")
                .Append(mL.Transforms.Conversion.MapValueToKey(outputColumnName: "MovieIdEncoded", inputColumnName: "MovieId"));

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "UserIdEncoded",
                MatrixRowIndexColumnName = "MovieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append(mL.Recommendation().Trainers.MatrixFactorization(options));

            ITransformer model = trainerEstimator.Fit(trainingDataView);
            return model;
        }

        public static void EvaluateModel(MLContext mL, IDataView testDataView, ITransformer model)
        {
            var prediction = model.Transform(testDataView);
            var metrics = mL.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError.ToString()}");
            Console.WriteLine($"RSquared: {metrics.RSquared.ToString()}");
        }

        public static void UseModelForSinglePrediction(MLContext mL, ITransformer model)
        {
            var engine = mL.Model
                .CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

            var sample = new MovieRating { UserId = 6, MovieId = 10 };
            var prediction = engine.Predict(sample);

            if (Math.Round(prediction.Score, 1) > 3.5)
            {
                Console.WriteLine($"Movie {sample.MovieId} is recommended for user {sample.UserId}");
            }
            else
            {
                Console.WriteLine($"Movie {sample.MovieId} is not recommended for user {sample.UserId}");
            }
        }

        public static void SaveModel(MLContext mL, DataViewSchema trainingSchema, ITransformer model)
        {
            var path = Path.Combine(Environment.CurrentDirectory, "Data", "model.zip");
            mL.Model.Save(model, trainingSchema, path);
        }
    }
}

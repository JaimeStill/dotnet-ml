using System;
using System.IO;
using Microsoft.ML;

namespace Regression
{
    class Program
    {
        static readonly string trainPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string testPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "model.zip");

        static void Main(string[] args)
        {
            MLContext mL = new MLContext(seed: 0);

            var model = Train(mL, trainPath);
            Evaluate(mL, model);
            TestSinglePrediction(mL, model);
        }

        public static ITransformer Train(MLContext mL, string dataPath)
        {
            IDataView dataView = mL.Data
                .LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

            var pipeline = mL.Transforms
                .CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mL.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mL.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mL.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mL.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripTime", "TripDistance", "PaymentTypeEncoded"))
                .Append(mL.Regression.Trainers.FastTree());

            var model = pipeline.Fit(dataView);

            return model;
        }

        private static void Evaluate(MLContext mL, ITransformer model)
        {
            IDataView dataView = mL.Data
                .LoadFromTextFile<TaxiTrip>(testPath, hasHeader: true, separatorChar: ',');

            var predictions = model.Transform(dataView);
            var metrics = mL.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine($"RSquared Score:          {metrics.RSquared:0.##}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine();
        }

        private static void TestSinglePrediction(MLContext mL, ITransformer model)
        {
            var engine = mL.Model
                .CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

            var sample = new TaxiTrip
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0
            };

            var prediction = engine.Predict(sample);

            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
        }
    }
}

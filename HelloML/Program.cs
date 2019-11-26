using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace HelloML
{
    class Program
    {
        public class HouseData
        {
            public float Size { get; set; }
            public float Price { get; set; }
        }

        public class Prediction
        {
            [ColumnName("Score")]
            public float Price { get; set; }
        }
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // Import or create training data
            HouseData[] houseData = 
            {
                new HouseData { Size = 1.1F, Price = 1.2F },
                new HouseData { Size = 1.9F, Price = 2.3F },
                new HouseData { Size = 2.8F, Price = 3.0F },
                new HouseData { Size = 3.4F, Price = 3.7F }
            };

            var trainingData = mlContext.Data.LoadFromEnumerable(houseData);

            // Specify data preparation and model training pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Size" })
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

            // train model
            var model = pipeline.Fit(trainingData);

            // Make a prediction
            var size = new HouseData() { Size = 2.5F };
            var price = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(size);

            Console.WriteLine($"Predicted price for size: {size.Size * 1000} sq ft = {price.Price * 100:C}k");

            HouseData[] testHouseData =
            {
                new HouseData { Size = 1.1F, Price = 0.98F },
                new HouseData { Size = 1.9F, Price = 2.1F },
                new HouseData { Size = 2.8F, Price = 2.9F },
                new HouseData { Size = 3.4F, Price = 3.6F }
            };

            var testTrainingData = mlContext.Data.LoadFromEnumerable(testHouseData);
            var testView = model.Transform(testTrainingData);

            var metrics = mlContext.Regression.Evaluate(testView, labelColumnName: "Price");

            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");
        }
    }
}

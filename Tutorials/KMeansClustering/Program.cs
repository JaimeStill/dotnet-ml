using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace KMeansClustering
{
    class Program
    {
        static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
        static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "model.zip");

        static void Main(string[] args)
        {
            var mL = new MLContext(seed: 0);

            var dataView = mL.Data
                .LoadFromTextFile<IrisData>(dataPath, hasHeader: false, separatorChar: ',');

            var pipeline = mL.Transforms
                .Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mL.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));

            var model = pipeline.Fit(dataView);

            mL.Model.Save(model, dataView.Schema, modelPath);

            var predictor = mL.Model
                .CreatePredictionEngine<IrisData, ClusterPrediction>(model);

            var prediction = predictor.Predict(TestIrisData.Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
        }
    }
}

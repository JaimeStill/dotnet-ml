using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace TransferLearning
{
    class Program
    {
        static readonly string assetsPath = Path.Combine(Environment.CurrentDirectory, "Data", "assets");
        static readonly string trainTagsTsv = Path.Combine(assetsPath, "inputs-train", "data", "tags.tsv");
        static readonly string predictImageListTsv = Path.Combine(assetsPath, "inputs-predict", "data", "image_list.tsv");
        static readonly string trainImagesFolder = Path.Combine(assetsPath, "inputs-train", "data");
        static readonly string predictImagesFolder = Path.Combine(assetsPath, "inputs-predict", "data");
        static readonly string predictSingleImage = Path.Combine(assetsPath, "inputs-predict-single", "data", "toaster3.jpg");
        static readonly string inceptionPb = Path.Combine(Environment.CurrentDirectory, "Data", "tensorflow_inception_graph.pb");
        static readonly string inputImageClassifierZip = Path.Combine(assetsPath, "inputs-predict", "imageClassifier.zip");
        static readonly string outputImageClassifierZip = Path.Combine(assetsPath, "outputs", "imageClassifier.zip");
        static string LabelToKey = nameof(LabelToKey);
        static string PredictedLabelValue = nameof(PredictedLabelValue);
        static void Main(string[] args)
        {
            var mL = new MLContext();
            var model = ReuseAndTuneInceptionModel(mL, trainTagsTsv, trainImagesFolder, inceptionPb, outputImageClassifierZip);
            ClassifyImages(mL, predictImageListTsv, predictImagesFolder, outputImageClassifierZip, model);
            ClassifySingleImage(mL, predictSingleImage, outputImageClassifierZip, model);
        }
        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }

        public static ITransformer ReuseAndTuneInceptionModel(MLContext mL, string dataLocation, string imagesFolder, string inputModelLocation, string outputModelLocation)
        {
            var data = mL.Data
                .LoadFromTextFile<ImageData>(path: dataLocation, hasHeader: false);

            var estimator = mL.Transforms
                .Conversion
                .MapValueToKey(outputColumnName: LabelToKey, inputColumnName: "Label")
                .Append(
                    mL.Transforms
                        .LoadImages(
                            outputColumnName: "input",
                            imageFolder: trainImagesFolder,
                            inputColumnName: nameof(ImageData.ImagePath)
                        )
                )
                .Append(
                    mL.Transforms
                        .ResizeImages(
                            outputColumnName: "input",
                            imageWidth: InceptionSettings.ImageWidth,
                            imageHeight: InceptionSettings.ImageHeight,
                            inputColumnName: "input"
                        )
                )
                .Append(
                    mL.Transforms
                        .ExtractPixels(
                            outputColumnName: "input",
                            interleavePixelColors: InceptionSettings.ChannelsLast,
                            offsetImage: InceptionSettings.Mean
                        )
                )
                .Append(
                    mL.Model
                        .LoadTensorFlowModel(inputModelLocation)
                        .ScoreTensorFlowModel(
                            outputColumnNames: new[] { "softmax2_pre_activation" },
                            inputColumnNames: new[] { "input" },
                            addBatchDimensionInput: true
                        )
                )
                .Append(
                    mL.MulticlassClassification
                        .Trainers
                        .LbfgsMaximumEntropy(
                            labelColumnName: LabelToKey,
                            featureColumnName: "softmax2_pre_activation"
                        )
                )
                .Append(
                    mL.Transforms
                        .Conversion
                        .MapKeyToValue(PredictedLabelValue, "PredictedLabel")
                );

            ITransformer model = estimator.Fit(data);

            var predictions = model.Transform(data);

            var imageData = mL.Data
                .CreateEnumerable<ImageData>(data, false, true);

            var imagePredictionData = mL.Data
                .CreateEnumerable<ImagePrediction>(predictions, false, true);

            DisplayResults(imagePredictionData);

            var metrics = mL.MulticlassClassification
                .Evaluate(predictions, labelColumnName: LabelToKey, predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"LogLoss: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss: {String.Join(" , ", metrics.PerClassLogLoss.Select(x => x.ToString()))}");
            Console.WriteLine();

            return model;
        }

        public static void ClassifyImages(MLContext mL, string dataLocation, string imagesFolder, string outputModelLocation, ITransformer model)
        {
            var imageData = ReadFromTsv(dataLocation, imagesFolder);

            var imageDataView = mL.Data
                .LoadFromEnumerable<ImageData>(imageData);

            var predictions = model.Transform(imageDataView);

            var imagePredictionData = mL.Data
                .CreateEnumerable<ImagePrediction>(predictions, false, true);

            DisplayResults(imagePredictionData);
        }

        public static void ClassifySingleImage(MLContext mL, string imagePath, string outputModelLocation, ITransformer model)
        {
            var imageData = new ImageData
            {
                ImagePath = imagePath
            };

            var predictor = mL.Model
                .CreatePredictionEngine<ImageData, ImagePrediction>(model);
            
            var prediction = predictor.Predict(imageData);

            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()}");
        }

        private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (var prediction in imagePredictionData)
            {
                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()}");
            }

            Console.WriteLine();
        }

        public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
        {
            return File.ReadAllLines(file)
                .Select(x => x.Split('\t'))
                .Select(x => new ImageData
                {
                    ImagePath = Path.Combine(folder, x[0])
                });
        }
    }
}

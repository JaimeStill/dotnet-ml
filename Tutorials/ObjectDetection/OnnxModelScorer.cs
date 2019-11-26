using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using ObjectDetection.DataStructures;
using ObjectDetection.YoloParser;

namespace ObjectDetection
{
    public class OnnxModelScorer
    {
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly MLContext mL;

        private IList<YoloBoundingBox> boundingBoxes = new List<YoloBoundingBox>();

        public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mL)
        {
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.mL = mL;
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 416;
            public const int imageWidth = 416;
        }

        public struct TinyYoloModelSettings
        {
            public const string ModelInput = "image";
            public const string ModelOutput = "grid";
        }

        private ITransformer LoadModel(string modelLocation)
        {
            Console.WriteLine("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight})");
            Console.WriteLine();

            var data = mL.Data.LoadFromEnumerable(new List<ImageNetData>());

            var pipeline = mL.Transforms
                .LoadImages(
                    outputColumnName: "image",
                    imageFolder: "",
                    inputColumnName: nameof(ImageNetData.ImagePath)
                )
                .Append(
                    mL.Transforms
                        .ResizeImages(
                            outputColumnName: "image",
                            imageWidth: ImageNetSettings.imageWidth,
                            imageHeight: ImageNetSettings.imageHeight,
                            inputColumnName: "image"
                        )
                )
                .Append(
                    mL.Transforms
                        .ExtractPixels(outputColumnName: "image")
                )
                .Append(
                    mL.Transforms
                        .ApplyOnnxModel(
                            modelFile: modelLocation,
                            outputColumnNames: new[] { TinyYoloModelSettings.ModelOutput },
                            inputColumnNames: new[] { TinyYoloModelSettings.ModelInput }
                        )
                );

            var model = pipeline.Fit(data);

            return model;
        }

        private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            Console.WriteLine($"Images location: {imagesFolder}");
            Console.WriteLine();
            Console.WriteLine("Identify the objects in the images");
            Console.WriteLine();

            IDataView scoredData = model.Transform(testData);
            IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>(TinyYoloModelSettings.ModelOutput);
            return probabilities;
        }

        public IEnumerable<float[]> Score(IDataView data)
        {
            var model = LoadModel(modelLocation);
            return PredictDataUsingModel(data, model);
        }
    }
}
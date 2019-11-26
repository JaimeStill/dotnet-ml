using System;
using System.IO;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using Microsoft.ML;
using ObjectDetection.YoloParser;
using ObjectDetection.DataStructures;

namespace ObjectDetection
{
    class Program
    {
        static readonly string assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        static readonly string modelPath = Path.Combine(assetsPath, "tiny_yolov2", "Model.onnx");
        static readonly string imagesFolder = Path.Combine(assetsPath, "images");
        static readonly string outputFolder = Path.Combine(imagesFolder, "output");
        static void Main(string[] args)
        {
            var mL = new MLContext();

            try
            {
                IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
                IDataView imageDataView = mL.Data.LoadFromEnumerable(images);

                var modelScorer = new OnnxModelScorer(imagesFolder, modelPath, mL);

                IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);

                YoloOutputParser parser = new YoloOutputParser();

                var boundingBoxes = probabilities
                    .Select(x => parser.ParseOutputs(x))
                    .Select(x => parser.FilterBoundingBoxes(x, 5, .5f));

                for (var i = 0; i < images.Count(); i++)
                {
                    string imageFileName = images.ElementAt(i).Label;
                    IList<YoloBoundingBox> detectedObjects = boundingBoxes.ElementAt(i);

                    DrawBoundingBox(imagesFolder, outputFolder, imageFileName, detectedObjects);
                    LogDetectedObjects(imageFileName, detectedObjects);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }

        private static void DrawBoundingBox(
            string inputImageLocation,
            string outputImageLocation,
            string imageName,
            IList<YoloBoundingBox> filteredBoundingBoxes
        )
        {
            Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName));

            var originalImageHeight = image.Height;
            var originalImageWidth = image.Width;

            foreach (var box in filteredBoundingBoxes)
            {
                var x = (uint)Math.Max(box.Dimensions.X, 0);
                var y = (uint)Math.Max(box.Dimensions.Y, 0);
                var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
                var height = (uint)Math.Max(originalImageHeight - y, box.Dimensions.Height);

                x = (uint) originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
                y = (uint) originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
                width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
                height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;

                string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";

                using (Graphics thumbnail = Graphics.FromImage(image))
                {
                    thumbnail.CompositingQuality = CompositingQuality.HighQuality;
                    thumbnail.SmoothingMode = SmoothingMode.HighQuality;
                    thumbnail.InterpolationMode = InterpolationMode.HighQualityBicubic;

                    Font drawFont = new Font("Arial", 12, FontStyle.Bold);
                    SizeF size = thumbnail.MeasureString(text, drawFont);
                    SolidBrush fontBrush = new SolidBrush(Color.Black);
                    Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                    Pen pen = new Pen(box.BoxColor, 3.2f);
                    SolidBrush colorBrush = new SolidBrush(box.BoxColor);

                    thumbnail.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                    thumbnail.DrawString(text, drawFont, fontBrush, atPoint);
                    thumbnail.DrawRectangle(pen, x, y, width, height);                    
                }
            }

            if (!Directory.Exists(outputImageLocation))
            {
                Directory.CreateDirectory(outputImageLocation);
            }

            image.Save(Path.Combine(outputImageLocation, imageName));
        }

        private static void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
        {
            Console.WriteLine($"Detected objects in {imageName}:");

            foreach (var box in boundingBoxes)
            {
                Console.WriteLine($"{box.Label} - Confidence Score: {box.Confidence}");
            }

            Console.WriteLine();
        }
    }
}

using Microsoft.ML.Data;

namespace TransferLearning
{
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }
    }

    public class ImagePrediction : ImageData
    {
        public float[] Score;
        public string PredictedLabelValue;
    }
}
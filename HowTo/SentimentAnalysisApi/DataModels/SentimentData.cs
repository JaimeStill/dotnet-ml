using Microsoft.ML.Data;

namespace SentimentAnalysisApi.DataModels
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string SentimentText { get; set; }

        [LoadColumn(1)]
        [ColumnName("Label")]
        public bool Sentiment { get; set; }
    }
}
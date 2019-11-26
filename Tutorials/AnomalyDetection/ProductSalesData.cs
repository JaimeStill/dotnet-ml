using Microsoft.ML.Data;

namespace AnomalyDetection
{
    public class ProductSalesData
    {
        [LoadColumn(0)]
        public string Month { get; set; }

        [LoadColumn(1)]
        public float NumSales { get; set; }
    }

    public class ProductSalesPrediction
    {
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
}
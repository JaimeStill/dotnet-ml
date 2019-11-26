using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;

namespace AnomalyDetection
{
    class Program
    {
        static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "product-sales.csv");
        const int docSize = 36;
        static void Main(string[] args)
        {
            var mL = new MLContext();

            var dataView = mL.Data
                .LoadFromTextFile<ProductSalesData>(path: dataPath, hasHeader: true, separatorChar: ',');

            DetectSpike(mL, docSize, dataView);
            DetectChangepoint(mL, docSize, dataView);           
        }

        static void DetectSpike(MLContext mL, int docSize, IDataView sales)
        {
            var spikeEstimator = mL.Transforms
                .DetectIidSpike(
                    outputColumnName: nameof(ProductSalesPrediction.Prediction),
                    inputColumnName: nameof(ProductSalesData.NumSales),
                    confidence: 95,
                    pvalueHistoryLength: docSize / 4
                );

            ITransformer spikeTransform = spikeEstimator.Fit(CreateEmptyDataView(mL));
            IDataView data = spikeTransform.Transform(sales);

            var predictions = mL.Data
                .CreateEnumerable<ProductSalesPrediction>(data, reuseRowObject: false);

            Console.WriteLine("Alert\tScore\tP-Value");

            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

                if (p.Prediction[0] == 1)
                {
                    results += " <-- Spike detected";
                }

                Console.WriteLine(results);
            }

            Console.WriteLine();
        }

        static void DetectChangepoint(MLContext mL, int docSize, IDataView sales)
        {
            var changePointEstimator = mL.Transforms
                .DetectIidChangePoint(
                    outputColumnName: nameof(ProductSalesPrediction.Prediction),
                    inputColumnName: nameof(ProductSalesData.NumSales),
                    confidence: 95,
                    changeHistoryLength: docSize / 4
                );

            ITransformer changePointTransform = changePointEstimator.Fit(CreateEmptyDataView(mL));
            IDataView data = changePointTransform.Transform(sales);

            var predictions = mL.Data
                .CreateEnumerable<ProductSalesPrediction>(data, reuseRowObject: false);

            Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");

            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";

                if (p.Prediction[0] == 1)
                {
                    results += " <-- alert is on, predicted changepoint";
                }

                Console.WriteLine(results);
            }

            Console.WriteLine();
        }

        static IDataView CreateEmptyDataView(MLContext mL)
        {
            IEnumerable<ProductSalesData> data = new List<ProductSalesData>();
            return mL.Data.LoadFromEnumerable(data);
        }
    }
}

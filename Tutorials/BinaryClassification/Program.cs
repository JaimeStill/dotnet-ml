using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace BinaryClassification
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            TrainTestData splitDataView = LoadData(mlContext);
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            Evaluate(mlContext, model, splitDataView.TestSet);
            UseModelWithSingleItem(mlContext, model);
            UseModelWithBatchItems(mlContext, model);
        }

        /*
            Loads the data, splits the loaded dataset into train and test datasets,
            and returns the split train and test data sets
        */
        public static TrainTestData LoadData(MLContext mLContext)
        {
            IDataView dataView = mLContext.Data
                .LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);

            /*
                Split the loaded dataset into train and test datasets and return them
                in the TrainTestData class. Specify the test percentage of data with
                the testFraction parameter. The default is 10%, in this case 20% is
                used to evaluate more data
            */
            TrainTestData splitDataView = mLContext.Data
                .TrainTestSplit(dataView, 0.2);

            return splitDataView;
        }

        public static ITransformer BuildAndTrainModel(MLContext mLContext, IDataView splitTrainSet)
        {
            /*
                Converts the text column SentimentText into a numeric key type
                Features column used by the machine learning algorithm and adds
                it as a new dataset column

                BinaryClassification task is used to categorize website comments as
                either positive or negative. This machine learning task is appended
                to the data transformation definitions using .Append

                The SdcaLogisticRegressionBinaryTrainer is the classification training
                algorithm. It is appended to the estimator and accepts the featurized
                SentimentText (Features) and the Label input parameters to learn from
                the historic data
            */
            var estimator = mLContext.Transforms
                .Text
                .FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mLContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            /*
                Fit the model to the splitTrainSet data and return the trained model
            */
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            return model;
        }

        /*
            Loads the test dataset, creates the BinaryClassification evaluator, Evaluates
            the model and creates metrics, and displays the metrics
        */
        public static void Evaluate(MLContext mLContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data ===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mLContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            /*
                Gets the accuracy of the model, which is the proportion of correct predictions
                in the test set
            */
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");

            /*
                The AreaUnderRocCurve metric indicates how confident the modl is correctly classifying
                the positive and negative classes. You want the AreaUnderRocCurve to be as close to
                one as possible
            */
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");

            /*
                The F1Score metric gets the model's F1 score, which is a measure between precision
                and recall. You want the F1Score to be as close to one as possible
            */
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        /*
            Creates a single comment of test data, predicts sentiment based on test data,
            combines test data and predictions for reporting, and displays the predicted
            results
        */
        private static void UseModelWithSingleItem(MLContext mLContext, ITransformer model)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mLContext.Model
                .CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");
            
            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine();
            Console.WriteLine("=============== End of Predictions ===============");
        }

        /*
            Creates batch test data, predicts sentiment based on test data, combines
            test data and predictions for reporting, and displays the predicted results
        */
        public static void UseModelWithBatchItems(MLContext mLContext, ITransformer model)
        {
            SentimentData[] sentiments =
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti"
                }
            };

            IDataView batchComments = mLContext.Data
                .LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            IEnumerable<SentimentPrediction> predictedResults = mLContext.Data
                .CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            Console.WriteLine();

            foreach (var prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");                
            }

            Console.WriteLine();
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}

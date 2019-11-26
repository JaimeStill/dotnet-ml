# Train a Machine Learning Model Using Cross Validation

* [Readme](./_readme.md)
* [The Data and Data Model](#the-data-and-data-model)
* [Prepare the Data](#prepare-the-data)
* [Train Model with Cross Validation](#train-model-with-cross-validation)
* [Evaluate the Model](#evaluate-the-model)

Learn how to use cross validation to train more robust machine learning models in <span>ML.NET</span>.

Cross-validation is a training and model evaluation technique that splits the data into several partitions and trains multiple algorithms on thetse partitions. This technique improves the robustness of the model by holding out data from the training process. In addition to improving performance on unseen observations, in data-constrained environments it can be an effective tool for training models with a smaller dataset.

## The Data and Data Model  
[Back to Top](#train-a-machine-learning-model-using-cross-validation)  

Given data from a file that has the following format:

```
Size (Sq. ft.), HistoricalPrice1 ($), HistoricalPrice2 ($), HistoricalPrice3 ($), Current Price ($)
620.00, 148330.32, 140913.81, 136686.39, 146105.37
550.00, 557033.46, 529181.78, 513306.33, 548677.95
1127.00, 479320.99, 455354.94, 441694.30, 472131.18
1120.00, 47504.98, 45129.73, 43775.84, 46792.41
```

The data can be modeled by a class like `HousingData`:

```cs
public class HousingData
{
    [LoadColumn(0)]
    public float Size { get; set; }

    [LoadColumn(1, 3)]
    [VectorType(3)]
    public float[] HistoricalPrices { get; set; }

    [LoadColumn(4)]
    [ColumnName("Label")]
    public float CurrentPrice { get; set; }
}
```

Load the data into an [IDataView](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview).

## Prepare the Data  
[Back to Top](#train-a-machine-learning-model-using-cross-validation)  

Pre-process the data before using it to build the machine learning model. In this sample, the `Size` and `HistoricalPrices` columns are combined into a single feature vector, which is output to a new column called `Features` using the [Concatenate](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transformextensionscatalog.concatenate) method. In addition to getting the data into the format expected by <span>ML.NET</span> algorithms, concatenating columns optimizes subsequent operations in the pipeline by applying the operation once for the concatenated column instead of each of the separate columns.

Once the columns are combined into a single vector, [NormalizeMinMax](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.normalizationcatalog.normalizeminmax) is applied to the `Features` column to get `Size` and `HistoricalPrices` in the same range between 0-1.

```cs
IEstimator<ITransformer> dataPrepEstimator = mlContext.Transforms
    .Concatenate("Features", new string[] { "Size", "HistoricalPrices" })
    .Append(
        mlContext.Transforms
            .NormalizeMinMax("Features")
    );

ITransformer dataPrepTransformer = dataPrepEstimator.Fit(data);

IDataView transformedData = dataPrepTransformer.Transform(data);
```

## Train Model with Cross Validation  
[Back to Top](#train-a-machine-learning-model-using-cross-validation)  

Once the data has been pre-processed, it's time to train the model. First, select the algorithm that most closely aligns with the machine learning task to be performed. Because the predicted value is a numerically continuous value, the task is regression. One of the regression algorithms implemented by <span>ML.NET</span> is the [StochasticDualCoordinateAscentCoordinator](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.sdcaregressiontrainer) algorithm. To train the model with cross-validation use the [CrossValidate](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.regressioncatalog.crossvalidate) method.

> Although this sample uses a linear regression model, CrossValidate is applicable to all other machine learning tasks in <span>ML.NET</span> except Anomaly Detection.

```cs
IEstimator<ITransformer> sdcaEstimator = mlContext.Regression
    .Trainers
    .Sdca();

var cvResults = mlContext.Regression
    .CrossValidate(transformedData, sdcaEstimator, numberOfFolds: 5);
```

[CrossValidate](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.regressioncatalog.crossvalidate) performs the following operations:
1. Partitions the data into a number of partitions equal to the value specified in the `numberOfFolds` parameter. The result of each partition is a [TrainTestData](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.dataoperationscatalog.traintestdata) object.
2. A model is trained on each of the partitions using the specified machine learning algorithm estimator on the training data set.
3. Each model's performance is evaluated using the [Evaluate](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.regressioncatalog.evaluate) method on the test data set.
4. The model along with its metrics are returned for each of the models.

The result stored in `cvResults` is a collection of [CrossValidationResult](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.traincatalogbase.crossvalidationresult-1) objects. The object includes the trained model as well as metrics which are both accessible from the [Model](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.traincatalogbase.crossvalidationresult-1.model) and [Metrics](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.traincatalogbase.crossvalidationresult-1.metrics) properties respectively. In this sample, the `Model` property is of type [ITransformer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.itransformer) and the `Metrics` property is of type [RegressionMetrics](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.regressionmetrics).

## Evaluate the Model  
[Back to Top](#train-a-machine-learning-model-using-cross-validation)  

Metrics for the different trained models can be accessed through the `Metrics` property of the individual [CrossValidationResult](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.traincatalogbase.crossvalidationresult-1) object. In this case, the [R-Squared metric](https://en.wikipedia.org/wiki/Coefficient_of_determination) is accessed and stored in the variable `rSquared`.

```cs
IEnumerable<double> rSquared = cvResults.Select(x => x.Metrics.RSquared);
```

If you inspect the contents of the `rSquared` variable, the output should be five values ranging from 0-1 where closer to 1 means best. Using metrics like R-Squared, select the models from best to wors performing. Then, select the top model to make predictions or perform additional operations with.

```cs
ITransformer topModel = cvResults
    .OrderByDescending(x => x.Metrics.RSquared)
    .Select(x => x.Model)
    .FirstOrDefault();
```
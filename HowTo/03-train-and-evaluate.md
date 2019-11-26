# Train and Evaluate a Model

* [Readme](./_readme.md)
* [Split Data for Training and Testing](#split-data-for-training-and-testing)
* [Prepare the Data](#prepare-the-data)
* [Train the Machine Learning Model](#train-the-machine-learning-model)
* [Extract Model Parameters](#extract-model-parameters)
* [Evaluate Model Quality](#evaluate-model-quality)

Learn how to build machine learning models, collect metrics, and measure performance with <span>ML.NET</span>. Although this sample trains a regression model, the concepts are applicable throughout a majority of the other algorithms.

## Split Data for Training and Testing  
[Back to Top](#train-and-evaluate-a-model)  

The goal of a machine learning model is to identify patterns within training data. These patterns are used to make predictions using new data.

Given the following data model:

```cs
public class HousingData
{
    [LoadColumn(0)]
    public float Size { get; set; }

    [LoadColumn(1, 3)]
    [VectorType(3)]
    public float HistoricalPrices { get; set; }

    [LoadColumn(4)]
    [ColumnName("Label")]
    public float CurrentPrice { get; set; }
}
```

Load the data into an [IDataView](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview):

```cs
HousingData[] housingData = new HousingData[]
{
    new HousingData
    {
        Size = 600f,
        HistoricalPrices = new float[] { 100000f, 125000f, 122000f },
        CurrentPrice = 170000f
    },
    new HousingData
    {
        Size = 1000f,
        HistoricalPrices = new float[] { 200000f, 250000f, 230000f },
        CurrentPrice = 225000f
    },
    new HousingData
    {
        Size = 1000f,
        HistoricalPrices = new float[] { 126000f, 130000f, 200000f },
        CurrentPrice = 195000f
    },
    new HousingData
    {
        Size = 850f,
        HistoricalPrices = new float[] { 150000f, 175000f, 210000f },
        CurrentPrice = 205000f
    },
    new HousingData
    {
        Size = 900f,
        HistoricalPrices = new float[] { 155000f, 190000f, 220000f },
        CurrentPrice = 210000f
    },
    new HousingData
    {
        Size = 550f,
        HistoricalPrices = new float[] { 99000f, 98000f, 130000f },
        CurrentPrice = 180000f
    }
};
```

Use the [TrainTestSplit](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.dataoperationscatalog.traintestsplit) method to split the data into train and test sets. The result will be a [TrainTestData](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.dataoperationscatalog.traintestdata) object which contains two [IDataView](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview) members, one for the train set and the other for the test set. The data split percentage is determined by the `testFraction` parameter. The snippet below is holding out 20 percent of the original data for the test set.  

```cs
DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data
    .TrainTestSplit(data, testFraction: 0.2);

IDataView trainData = dataSplit.TrainSet;
IDataView testData = dataSplit.TestSet;
```

## Prepare the Data  
[Back to Top](#train-and-evaluate-a-model)  

The data needs to be pre-processed before before training a machine learning model. More information on data preparation can be found on the [data prep how-to article](https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/prepare-data-ml-net) as well as the [transforms page](https://docs.microsoft.com/en-us/dotnet/machine-learning/resources/transforms).

<span>ML.NET</span> algorithms have constraints on input column types. Additionally, default values are used for input and output column names when no values are specified.

### Working with expected column types

The machine learning algorithms in <span>ML.NET</span> expect a float vector of known size as input. Apply the [VectorType](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.vectortypeattribute) attribute to your data model when all of the data is already in numerical format and is intended to be processed together (i.e. image pixels).

If the data is not all numerical and you want to apply different data transformations of each of the columns individually, use the [Concatenate](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transformextensionscatalog.concatenate) method after all of the columns have been processed to combine all of the individual columns into a single feature vector that is output to a new column.

The following snippet combines the `Size` and `HistoricalPrices` columns into a single feature vector that is output to a new column called `Features`. Because there is a difference in scales, [NormalizeMinMax](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.normalizationcatalog.normalizeminmax) is applied to the `Features` column to normalize the data.

```cs
IEstimator<ITransformer> dataPrepEstimator = mlContext.Transforms
    .Concatenate("Features", "Size", "HistoricalPrices")
    .Append(
        mlContext.Transforms.NormalizeMinMax("Features")
    );

ITransformer dataPrepTransformer = dataPrepEstimator.Fit(trainData);

IDataView transformedTrainingData = dataPrepTransformer.Transform(trainData);
```

### Working with default column names

<span>ML.NET</span> algorithms use default column names when none are specified. All trainers have a parameter called `featureColumnName` for the inputs of the algorithm and wehn applicable they also have a parameter for the expected value called `labelColumnName`. By default those values are `Features` and `Label` respectively.

By using the [Concatenate](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transformextensionscatalog.concatenate) method during pre-processing to create a new column called `Features`, there is no need to specify the feature column name in the parameters of the algorithm since it already exists in the pre-processed `IDataView`. The label column is `CurrentPrice`, but since the [ColumnName](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.columnnameattribute) attribute is used in the data mode, <span>ML.NET</span> renames the `CurrentPrice` column to `Label` which removes the need to provide the `labelColumnName` parameter to the machine learning algorithm estimator.

If you don't want to use the default column names, pass in the names of the feature and label columns as parameters when defining the machine learning algorithm estimator as demonstrated by the subsequent snippet:

```cs
var userDefinedColumnSdcaEstimator = mlContext.Regression
    .Trainers
    .Sdca(
        labelColumnName: "MyLabelColumnName",
        featureColumnName: "MyFeatureColumnName"
    );
```

## Train the Machine Learning Model  
[Back to Top](#train-and-evaluate-a-model)  

Once the data is pre-processed, use the [Fit](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.trainerestimatorbase-2.fit) method to train the machine learning model with the [StochasticDualCoordinateAscent](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.sdcaregressiontrainer) regression algorithm.

```cs
var sdcaEstimator = mlContext.Regression.Trainers.Sdca();
var trainedModel = sdcaEstimator.Fit(transformedTrainingData);
```

## Extract Model Parameters  
[Back to Top](#train-and-evaluate-a-model)  

After the model has been trained, extract the learned [ModelParameters](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.modelparametersbase-1) for inspection or re-training. The [LinearRegressionModelParameters](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.linearregressionmodelparameters) provide the bias and learned coefficients or weights of the trained model.

```cs
var trainedModelParameters = trainedModel.Model as LinearRegressionModelParameters;
```

> Other models have parameters that are specific to their tasks. For example, the [K-Means algorithm](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.kmeanstrainer) puts data into cluster-based or centroids and the [KMeansModelParameters](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.kmeansmodelparameters) contains a property that stores these learned centroids. To learm more, visit the [Microsoft.ML.Trainers API Documentation](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers) and look for classes that contain `ModelParameters` in their name.

## Evaluate Model Quality  
[Back to Top](#train-and-evaluate-a-model)  

To help choose the best performing model, it is essential to evaluate its performance on test data. Use the [Evaluate](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.regressioncatalog.evaluate) method to measure various metrics for the trained model.

> The `Evaluate` method produces different metrics depending on which machine learning task was performed. For more details, visit the [Microsoft.ML.Data API Documentation](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data) and look for classes that contain `Metrics` in their name.

```cs
IDataView transformedTestData = dataPrepTransformer.Transform(testData);

IDataView testDataPredictions = trainedModel.Transform(transformedTestData);

RegressionMetrics trainedModelMetrics = mlContext.Regression
    .Evaluate(testDataPredictions);

double rSquared = trainedModelMetrics.RSquared;
```

In the previous code sample:
1. Test data set is pre-processed using the data preparation transforms previously defined.
2. The trained machine learning model is used to make predictions on the test data.
3. In the `Evaluate` method, the values in the `CurrentPrice` column of the test data set are compared against the `Score` column of the newly output predictions to calculate the metrics for the regression model, one of which, R-Squared, is stored in the `rSquared` variable.

> In this small example, the R-Squared is a number not in the range of 0-1 because of the limited size of the data. In a real-world scenario, you would expect to see a value between 0 and 1.
# Explain Model Predictions Using Permutation Feature Importance

* [Readme](./_readme.md)
* [Load the Data](#load-the-data)
* [Train the Model](#train-the-model)
* [Explain the Model With Permutation Feature Importance](#explain-the-model-with-permutation-feature-importance)

Learn how to explain <span>ML.NET</span> machine learning model predictions by understanding the contribution features have to predictions using Permutation Feature Importance (PFI).

Machine learning models are often thought of as black boxes that take inputs and generate an output. The intermediate steps or interactions among the features that influence the output are rarely understood. As machine learning is introduced into more aspects of everyday life such as healthcare, it's of utmost importance to understand why a machine learning model makes the decisions it does. For example, if diagnoses are made by a machine learning model, healthcare professionals need a way to look into the factors that went into making that diagnosis. Providing the right diagnosis could make a great difference on whether a patient has a speedy recovery or not. Therefore the higher the level of explainability in a model, the greater confidence healthcare professionals have to accept or reject the decisions made by the model.

Various techniques are used to explain models, one of which is PFI. PFI is a technique used to explain classification and regression models that is inspired by [Breiman's *Random Forests* paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)(see section 10). At a high level, the way it works is by randomly shuffling data one feature at a time for the entire dataset and calculating how much the performance metric of interest decreases. The larger the change, the more important that feature is.

Additionally, by highlighting the most important features, model builders can focus on using a subset of more meaningful features which can potentially reduce noise and training time.

## Load the Data  
[Back to Top](#explain-model-predictions-using-permutation-feature-importance)  

The features in the dataset being used for this sample are in columns 1-12. The goal is to predict `Price`.

Column | Feature | Description
-------|---------|------------
1 | `CrimeRate` | Per capita crime rate
2 | `ResidentialZones` | Residential zones in town
3 | `CommercialZones` | Non-residential zones in town
4 | `NearWater` | Proximity to body of water
5 | `ToxicWasteLevels` | Toxicity levels (PPM)
6 | `AverageRoomNumber` | Average number of rooms in house
7 | `HomeAge` | Age of home
8 | `BusinessCenterDistance` | Distance to nearest business district
9 | `HighwayAccess` | Proximity to highways
10 | `TaxRate` | Property tax rate
11 | `StudentTeacherRatio` | Ratio of students to teachers
12 | `PercentPopulationBelowPoverty` | Percent of population living below poverty
13 | `Price` | Price of the home

A sample of the dataset is shown below:

```
1,24,13,1,0.59,3,96,11,23,608,14,13,32
4,80,18,1,0.37,5,14,7,4,346,19,13,41
2,98,16,1,0.25,10,5,1,8,689,13,36,12
```

The data in this sample can be modeled by a class like `HousingPriceData`:

```cs
class HousingPriceData
{
    [LoadColumn(0)]
    public float CrimeRate { get; set; }

    [LoadColumn(1)]
    public float ResidentialZones { get; set; }

    [LoadColumn(2)]
    public float CommercialZones { get; set; }

    [LoadColumn(3)]
    public float NearWater { get; set; }

    [LoadColumn(4)]
    public float ToxicWasteLevels { get; set; }

    [LoadColumn(5)]
    public float AverageRoomNumber { get; set; }

    [LoadColumn(6)]
    public float HomeAge { get; set; }

    [LoadColumn(7)]
    public float BusinessCenterDistance { get; set; }

    [LoadColumn(8)]
    public float HighwayAccess { get; set; }

    [LoadColumn(9)]
    public float TaxRate { get; set; }

    [LoadColumn(10)]
    public float StudentTeacherRatio { get; set; }

    [LoadColumn(11)]
    public float PercentPopulationBelowPoverty { get; set; }

    [LoadColumn(12)]
    [ColumnName("Label")]
    public float Price { get; set; }
}
```

Load the data into an [IDataView](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview).

## Train the Model  
[Back to Top](#explain-model-predictions-using-permutation-feature-importance)  

The code sample below illustrates the process of training a linear regression model to predict house prices.

```cs
string[] featureColumnNames = data.Schema
    .Select(x => x.Name)
    .Where(x => x != "Label")
    .ToArray();

IEstimator<ITransformer> dataPrepEstimator = mlContext.Transforms
    .Concatenate("Features", featureColumnNames)
    .Append(
        mlContext.Transforms.NormalizeMinMax("Features");
    );

ITransformer dataPrepTransformer = dataPrepEstimator.Fit(data);

IDataView preprocessedTrainData = dataPrepTransformer.Transform(data);

var sdcaEstimator = mlContext.Regression.Trainers.Sdca();
var sdcaModel = sdcaEstimator.Fit(preprocessedTrainData);
```

## Explain the Model with Permutation Feature Importance  
[Back to Top](#explain-model-predictions-using-permutation-feature-importance)  

In <span>ML.NET</span> use the [PermutationFeatureImportance](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.permutationfeatureimportanceextensions) method for your respective task.

```cs
ImmutableArray<RegressionMetricsStatistics> pfi = mlContext.Regression
    .PermutationFeatureImportance(
        sdcaModel,
        preprocessedTrainData,
        permutationCount: 3
    );
```

The result of using [PermutationFeatureImportance](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.permutationfeatureimportanceextensions) on the training dataset is an [ImmutableArray](https://docs.microsoft.com/en-us/dotnet/api/system.collections.immutable.immutablearray) of [RegressionMetricsStatistics](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.regressionmetricsstatistics) objects. [RegressionMetricsStatistics](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.regressionmetricsstatistics) provides summary statistics like mean and standard deviation for multiple observations of [RegressionMetrics](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.regressionmetrics) equal to the number of permutations specified by the `permutationCount` parameter.

The importance, or in this case, the absoluate average decrease in R-squared metric calculated by [PermutationFeatureImportance](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.permutationfeatureimportanceextensions) can then be ordered from most important to least important.

```cs
var featureImportanceMetrics = permutationFeatureImportance
    .Select((x, i) => new { i, x.RSquared })
    .OrderByDescending(x => Math.Abs(x.RSquared.Mean));

Console.WriteLine("Feature\tPFI");

foreach (var feature in featureImportanceMetrics)
{
    Console.WriteLine(
        $"{featureColumnNames[feature.index], -20}\t{feature.RSquared.Mean:F6}"
    );
}
```

Printing the values for each of the features in the `featureImportanceMetrics` would generate output similar to that below. Keep in mind that you should expect to see different results because these values vary based on the data that they are given.

Feature | Change to R-Squared
--------|--------------------
`HighwayAccess` | -0.042731
`StudentTeacherRatio` | -0.012730
`BusinessCenterDistance` | -0.010491
`TaxRate` | -0.008545
`AverageRoomNumber` | -0.003949
`CrimeRate` | -0.003665
`CommercialZones` | 0.002749
`HomeAge` | -0.002426
`ResidentialZones` | -0.000203
`NearWater` | 0.000203
`PercentPopulationLivingBelowPoverty` | 0.000031
`ToxicWasteLevels` | -0.000019

Taking a look at the five most important features for this dataset, the price of a house predicted by this model is influenced by its proximity to highways, student teacher ratio of schools in the area, proximity to major employment centers, property tax rate and average number of rooms in the home.
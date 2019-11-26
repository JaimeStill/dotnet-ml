# Re-train a Model

* [Readme](./_readme.md)
* [Load Pre-trained Model](#load-pre-trained-model)
* [Extract Pre-trained Model Parameters](#extract-pre-trained-model-parameters)
* [Re-train Model](#re-train-model)
* [Compare Model Parameters](#compare-model-parameters)

Learn how to retrain a machine learning model in <span>ML.NET</span>.

The world and the data around it change at a constant pace. As such, models need to change and update as well. <span>ML.NET</span> provides functionality for re-training models using learned model parameters as a starting point to continually build on previous experience rather than starting from scratch every time.

The following algorithms are re-trainable in <span>ML.NET</span>:
* [AveragedPerceptronTrainer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.averagedperceptrontrainer)
* [FieldAwareFactorizationMachineTrainer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.fieldawarefactorizationmachinetrainer)
* [LbfgsLogsitcRegressionBinaryTrainer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.lbfgslogisticregressionbinarytrainer)
* [LbfgsMaximumEntropyMulticlassTrainer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.lbfgsmaximumentropymulticlasstrainer)
* [LinearSvmTrainer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.linearsvmtrainer)
* [OnlineGradientDescentTrainer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.onlinegradientdescenttrainer)
* [SgdCalibratedTrainer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.sgdcalibratedtrainer)
* [SgdNonCalibratedTrainer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.sgdnoncalibratedtrainer)
* [SymbolicSgdLogisticRegressionBinaryTrainer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.symbolicsgdlogisticregressionbinarytrainer)

## Load Pre-trained Models  
[Back to Top](#re-train-a-model)  

First, load the pre-trained model into your application.

```cs
MLContext mlContext = new MLContext();

DataViewSchema pipelineSchema, modelSchema;

ITransformer pipeline = mlContext.Model
    .Load("pipeline.zip", out pipelineSchema);

ITransformer trainedModel = mlContext.Model
    .Load("model.zip", out modelSchema);
```

## Extract Pre-trained Model Parameters  
[Back to Top](#re-train-a-model)  

Once the model is loaded, extract the learned model parameters by accessing the [Model](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.predictiontransformerbase-1.model) property of the pre-trained model. The pre-trained model was trained using the linear regression model [OnlineGradientDescentTrainer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.onlinegradientdescenttrainer) which creates a [RegressionPredictionTransformer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.regressionpredictiontransformer-1) that outputs [LinearRegressionModelParameters](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.linearregressionmodelparameters). These linear regression model parameters contain the learned bias and weights or coefficients of the model. These values will be used as a starting point for the new re-trained model.

```cs
LinearRegressionModelParameters = originalModelParameters = ((ISingleFeaturePredictionTransformer<object>)trainedModel).Model as LinearRegressionModelParameters;
```

## Re-train Model  
[Back to Top](#re-train-a-model)  

The process for retraining a model is no different than that of training a model. The only difference is, the [Fit](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.onlinelineartrainer-2.fit) method in addition to the data also takes as input the original learned model parameters and uses them as a starting point in the re-training process.

```cs
HousingData[] housingData = new HousingData[]
{
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

IDataView newData = mlContext.Data.LoadFromEnumerable<HousingData>(housingData);

IDataView transformedNewData = dataPrepPipeline.Transform(newData);

RegressionPredictionTransformer<LinearRegressionModelParameters> retrainedModel =
    mlContext.Regression
        .Trainers
        .OnlineGradientDescent()
        .Fit(transformedNewData, originalModelParameters);
```

## Compare Model Parameters  
[Back to Top](#re-train-a-model)  

How do you know if re-training actually happened? One way would be to compare whether the re-trained model's parameters are different than those of the original model. The code sample below compares the original agains the re-trained model weights and outputs them to the console.

```cs
LinearRegressionModelParameters retrainedModelParameters = 
    retrainedModel.Model as LinearRegressionModelParameters;

var weightDiffs = originalParameters.Weights
    .Zip(
        retrainedModelParameters.Weights,
        (original, retrained) => original - retrained
    ).ToArray();

Console.WriteLine("Original | Retrained | Difference");

for (var i = 0; i < weightDiffs.Count(); i++)
{
    Console.WriteLine($"{originalModelParameters.Weights[i]} | {retrainedModelParameters.Weights[i]} | {weightDiffs[i]}");
}
```

The table below shows what the output might look like.

Original | Retrained | Difference
---------|-----------|-----------
33039.86 | 56293.76 | -23253.9
29099.14 | 49586.03 | -20486.89
28938.38 | 48609.23 | -19670.85
30484.02 | 53745.43 | -23261.41
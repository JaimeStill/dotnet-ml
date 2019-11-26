# Save and Load Trained Models

* [Readme](./_readme.md)
* [Save a Model Locally](#save-a-model-locally)
* [Load a Model Stored Locally](#load-a-model-stored-locally)
* [Load a Model Stored Remotely](#load-a-model-stored-remotely)
* [Working with Separate Data Preparation and Model Pipelines](#working-with-separate-data-preparation-and-model-pipelines)

Learn how to save and load trained models in your application.

Throught the model building process, a model lives in memory and is accessible throughout your application's lifecycle. However, once the application stops running, if the model is not saved somewhere locally or remotely, it's no longer accessible. Typically models are used at some point after training in other applications either for inference or re-training. Therefore, it's important to store the model. Save and load models using the steps described in subsequent sections of this document when using data preparation and model training pipelines like the one detailed below. Although this sample uses a linear regression model, the same process applies to other <span>ML.NET</span> algorithms.

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
    }
};

MLContext mlContext = new MLContext();

IDataView data = mlContext.Data
    .LoadFromEnumerable<HousingData>(housingData);

EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> estimator = 
    mlContext.Transforms
        .Concatenate("Features", new string[] { "Size", "HistoricalPrices" })
        .Append(
            mlContext.Transforms.NormalizeMinMax("Features")
        )
        .Append(
            mlContext.Regression.Trainers.Sdca()
        );

ITransformer trainedModel = pipelineEstimator.Fit(data);

mlContext.Model.Save(trainedModel, data.Schema, "model.zip");
```

Because most models and data preparation pipelines inherit from thet same set of classes, the save and load method signatures for those components is the same. Depending on your use case, you can either combine the data preparation pipeline and model into a single [EstimatorChain](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.transformerchain-1) which would output a single [ITransformer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.itransformer) or separate them thus creating a separate [ITransformer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.itransformer) for each.

## Save a Model Locally  
[Back to Top](#save-and-load-trained-models)  

When saving a model you need two things:
1. The [ITransformer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.itransformer) of the model.
2. The [DataViewSchema](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.dataviewschema) of the [ITransformer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.itransformer)'s expected input.

After training the model, use the [Save](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.modeloperationscatalog.save) method to save the trained model to a file called `model.zip` using the `DataViewSchema` of the input data.

```cs
mlContext.Model.Save(trainedModel, data.Schema, "model.zip");
```

## Load a Model Stored Locally  
[Back to Top](#save-and-load-trained-models)  

Models stored locally can be used in other processes or applications like `ASP.NET Core` and `Serverless Web Applications`.

In a separate application or process, use the [Load](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.modeloperationscatalog.load) method along with the file path to get the trained model into your application.

```cs
DataViewSchema modelSchema;

ITransformer trainedModel = mlContext.Model.Load("model.zip", out modelSchema);
```

## Load a Model Stored Remotely  
[Back to Top](#save-and-load-trained-models)  

To load data preparation pipelines and models stored in a remote location in your application, use a [Stream](https://docs.microsoft.com/en-us/dotnet/api/system.io.stream) instead of a file path in the [Load](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.modeloperationscatalog.load) method.

```cs
MLContext mlContext = new MLContext();

DataViewSchema modelSchema;
ITransformer trainedModel;

using (HttpClient client = new HttpClient())
{
    Stream modelFile = await client.GetStreamAsync("<YOUR-REMOTE-FILE-LOCATION>");
    trainedModel = mlContext.Model.Load(modelFile, out modelSchema);
}
```

## Working with Separate Data Preparation and Model Pipelines  
[Back to Top](#save-and-load-trained-models)  

> Working with separate data preparation and model training pipelines is optional. Separation of pipelines makes it easier to inspect the learned model parameters. For predictions, it's easier to save and load a single pipeline that includes the data preparation and model training operations.

When working with separate data preparation pipelines and models, the same process as single pipelines applies; except now both pipelines need to be saved and loaded simultaneously.

Given separate data preparation and model training pipelines:

```cs
IEstimator<ITransformer> dataPrepEstimator = mlContext.Transforms
    .Concatenate("Features", new string[] { "Size", "HistoricalPrices" })
    .Append(
        mlContext.Transforms.NormalizeMinMax("Features")
    );

ITransformer dataPrepTransformer = dataPrepEstimator.Fit(data);

var sdcaEstimator = mlContext.Regression.Trainers.Sdca();

IDataView transformedData = dataPrepTransformer.Transform(data);

RegressionPredictionTransformer<LinearRegressionModelParameters> trainedModel =
    sdcaEstimator.Fit(transformedData);
```

### Save data preparation pipeline and trained model

To save both the preparation pipeline and trained model, use the following commands:

```cs
mlContext.Model.Save(dataPrepTransformer, data.Schema, "pipeline.zip");

mlContext.Model.Save(trainedModel, transformedData.Schema, "model.zip");
```

### Load data preparation pipeline and trained model

In a separate process or application, load the data preparation pipeline and trained model simultaneously as follows:

```cs
MLContext mlContext = new MLContext();

DataViewSchema dataPrepPipelineSchema, modelSchema;

ITransformer dataPrepPipeline = mlContext.Model
    .Load("pipeline.zip", out dataPrepPipelineSchema);

ITransformer trainedModel = mlContext.Model
    .Load("model.zip", out modelSchema);
```
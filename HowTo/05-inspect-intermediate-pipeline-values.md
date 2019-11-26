# Inspect Intermediate Data During Processing

* [Readme](./_readme.md)
* [Convert IDataView to IEnumerable](#convert-idataview-to-ienumerable)
* [Assessing Specific Indices with IEnumerable](#assessing-specific-indices-with-ienumerable)
* [Inspect Values in a Single Column](#inspect-values-in-a-single-column)
* [Inspect IDataView Values One Row at a Time](#inspect-idataview-values-one-row-at-a-time)
* [Preview Result of Pre-processing or Training on a Subset of the Data](#preview-result-of-pre-processing-or-training-on-a-subset-of-the-data)

Learn how to inspect intermediate data during loading, processing, and model training steps in <span>ML.NET</span>. Intermediate data is the output of each stage in the machine learning pipeline.

Intermediate data like the one represented below which is loaded into an [IDataView](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview) can be inspected in various ways in <span>ML.NET</span>.

```cs
HousingData[] housingData = new HousingData[]
{
    new HousingData
    {
        Size = 600f,
        HistoricalPrices = new float[] { 100000f ,125000f ,122000f },
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
        HistoricalPrices = new float[] { 150000f,175000f,210000f },
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

## Convert IDataView to IEnumerable  
[Back to Top](#inspect-intermediate-data-during-processing)  

One of the quickest ways to inspect an [IDataView](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview) is to convert it to an [IEnumerable](https://docs.microsoft.com/en-us/dotnet/api/system.collections.generic.ienumerable-1). To convert an [IDataView](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview) to [IEnumerable](https://docs.microsoft.com/en-us/dotnet/api/system.collections.generic.ienumerable-1) use the [CreateEnumerable](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.dataoperationscatalog.createenumerable) method.

To optimize performance, set `reuseRowObject` to `true`. Doing so will lazily populate the same object with the data of the current row as it's being evaluated as opposed to creating a new object for each row in the dataset.

```cs
IEnumerable<HousingData> housingDataEnumerable = mlContext.Data
    .CreateEnumerable<HousingData>(data, reuseRowObject: true);

foreach (HousingData row in housingDataEnumerable)
{
    Console.WriteLine(row.Size);
}
```

## Assessing Specific Indices with IEnumerable  
[Back to Top](#inspect-intermediate-data-during-processing)  

If you only need access to a portion of the data or specific indices, use [CreateEnumerable](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.dataoperationscatalog.createenumerable) and set the `reuseRowObject` parameter value to `false` so a new object is created for each of the requested rows in the dataset. Then, convert [IEnumerable](https://docs.microsoft.com/en-us/dotnet/api/system.collections.generic.ienumerable-1) to an array or list.

> Converting the result of [CreateEnumerable](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.dataoperationscatalog.createenumerable) to an array or list will load all the requested [IDataView](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview) rows into memory which may affect performance.

Once the collection has been created, you can perform operations on the data. The code snippet below takes the first three rows in the dataset and calculates the average current price.

```cs
HousingData[] housingDataArray = mlContext.Data
    .CreateEnumerable<HousingData>(data, reuseRowObject: false)
    .Take(3)
    .ToArray();

HousingData first = housingDataArray[0];
HousingData second = housingDataArray[1];
HousingData third = housingDataArray[2];

float averageCurrentPrice = (first.CurrentPrice + second.CurrentPrice + third.CurrentPrice) / 3;
```

## Inspect Values in a Single Column  
[Back to Top](#inspect-intermediate-data-during-processing)  

At any point in the model building process, values in a single column of an [IDataView](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview) can be accessed using the [GetColumn](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.columncursorextensions.getcolumn) method. The [GetColumn](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.columncursorextensions.getcolumn) method returns all of the values in a single column as an [IEnumerable](https://docs.microsoft.com/en-us/dotnet/api/system.collections.generic.ienumerable-1).

```cs
IEnumerable<float> sizeColumn = data.GetColumn<float>("Size").ToList();
```

## Inspect IDataView Values One Row at a Time  
[Back to Top](#inspect-intermediate-data-during-processing)  

[IDataView](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview) is lazily evaluated. To iterate over the rows of an [IDataView](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview) without converting to an [IEnumerable](https://docs.microsoft.com/en-us/dotnet/api/system.collections.generic.ienumerable-1) as demonstrated in previous sections of this document, create a [DataViewRowCursor](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.dataviewrowcursor) by using the [GetRowCursor](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview.getrowcursor) method and passing in the [DataViewSchema](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.dataviewschema) of your [IDataView](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview) as a parameter. Then, to iterate over rows, use the [MoveNext](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.dataviewrowcursor.movenext) cursor method along with [ValueGetter](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.valuegetter-1) delegates to extract the respective values form each of the columns.

> For performance purposes, vectors in <span>ML.NET</span> use [VBuffer](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.vbuffer-1) instead of native collection types (that is, `Vector`, `float[]`).

```cs
DataViewSchema columns = data.Schema;

using (DataViewRowCursor cursor = data.GetRowCursor(columns))
{
    float size = default;
    VBuffer<float> historicalPrices = default;
    float currentPrice = default;

    ValueGetter<float> sizeDelegate = cursor.GetGetter<float>(columns[0]);
    ValueGetter<VBuffer<float>> historicalPriceDelegate = cursor.GetGetter<VBuffer<float>>(columns[1]);
    ValueGetter<float> currentPriceDelegate = cursor.GetGetter<float>(columns[2]);

    while (cursor.MoveNext())
    {
        sizeDelegate.Invoke(ref size);
        historicalPriceDelegate.Invoke(ref historicalPrices);
        currentPriceDelegate.Invoke(ref currentPrice);
    }
}
```

## Preview Result of Pre-processing or Training on a Subset of the Data  
[Back to Top](#inspect-intermediate-data-during-processing)  

> Do not use `Preview` in production code because it is intended for debugging and may reduce performance.

The model building process is experimental and iterative. To preview what data would look like after pre-processing or training a machine learning model on a subset of the data, use the [Preview](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.debuggerextensions.preview) method which returns a [DataDebuggerPreview](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.datadebuggerpreview). The result is an object with `ColumnView` and `RowView` properties which are both an [IEnumerable](https://docs.microsoft.com/en-us/dotnet/api/system.collections.generic.ienumerable-1) and contain the values in a particular column or row. Specify the number of rows to apply the transformation to with the `maxRows` parameter.
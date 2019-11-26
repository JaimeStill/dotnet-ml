# Load Data from Files and Other Sources

* [Readme](./_readme.md)  
* [Create the Data Model](#create-the-data-model)
* [Load Data From a Single File](#load-data-from-a-single-file)
* [Load Data From Multiple Files](#load-data-from-multiple-files)
* [Load Data From Other Sources](#load-data-from-other-sources)

This how-to shows you how to load data for processing and training into <span>ML.NET</span>. The data is originally stored in files or other data sources such as databases, JSON, XML or in-memory collections.

## Create the Data Model  
[Back to Top](#load-data-from-files-and-other-sources)  

<span>ML.NET</span> enables you to define data models via classes. For example, given the following input data:

```
Size (Sq. ft.), HistoricalPrice1 ($), HistoricalPrice2 ($), HistoricalPrice3 ($), Current Price ($)
700, 100000, 3000000, 250000, 500000
1000, 600000, 400000, 650000, 700000
```

Create a data model that represents the snippet below:

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

### Annotating the data model with column attributes

Attributes give <span>ML.NET</span> more information about the data model and the data source.

The [LoadColumn](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.loadcolumnattribute) attribute specifies your properties' column indices.

> `LoadColumn` is only required when loading data from a file.

Load columns as:
* Individual columns like `Size` and `CurrentPrices` in the `HousingData` class.
* Multiple columns at a time in the form of a vector like `HistoricalPrices` in the `HousingData` class.

If you have a vector property, apply the [VectorType](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.vectortypeattribute) attribute to the property in your data model. It's important to note that all of the elements in the vector need to be the same type. Keeping the columns separated allows for ease and flexibility of feature engineering, but for a very large number of columns, operating on the individual columns causes an impact on training speed.

<span>ML.NET</span> operates through column names. If you want to change the name of a column to something other than the property name, use the [ColumnName](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.columnnameattribute) attribute. When creating in-memory objects, you still create objects using the property name. However, for data processing and building machine learning models, <span>ML.NET</span> overrides and references the property with the value provided in the [ColumnName](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.columnnameattribute) attribute.

## Load Data From a Single File  
[Back to Top](#load-data-from-files-and-other-sources)  

To load data from a file, use the [LoadFromTextFile](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.textloadersavercatalog.loadfromtextfile) method along with the data model for the data to be loaded. Since `separatorChar` parameter is tab-delimeted by default, change it for your data file as needed. If your file has a header, set the `hasHeader` parameter to `true` to ignore the first line in the file and begin to load data from the second line.

```cs
MLContext mlContext = new MLContext();

IDataView data = mlContext.Data
    .LoadFromTextFile<HousingData>(
        "my-data-file.csv",
        separatorChar: ',',
        hasHeader: true
    );
```

## Load Data From Multiple Files  
[Back to Top](#load-data-from-files-and-other-sources)  

In the event that your data is stored in multiple files, as long as the data schema is the same, <span>ML.NET</span> allows you to load data from multiple files that are either in the same directory or multiple directories.

### Load from files in a single directory

When all of your data files are in the same directory, use wildcards in the [LoadFromTextFile](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.textloadersavercatalog.loadfromtextfile) method.

```cs
MLContext mlContext = new MLContext();

IDataView data = mlContext.Data
    .LoadFromTextFile<HousingData>(
        "Data/*",
        separatorChar: ',',
        hasHeader: true
    );
```

### Load from files in multiple directories

To load data from multiple directories, use the [CreateTextLoader](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.textloadersavercatalog.createtextloader) method to create a [TextLoader](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.data.textloader). Then, use the [TextLoader.Load](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.dataloaderextensions.load) method and specify the individual paths (wildcards can't be used).

```cs
MLContext mlContext = new MLContext();

TextLoader textLoader = mlContext.Data
    .CreateTextLoader<HousingData>(
        separatorChar: ',',
        hasHeader: true
    );

IDataView data = textLoader.Load(
    "DataFolder/SubFolder1/1.txt",
    "DataFolder/SubFolder2/1.txt"
);
```

## Load Data From Other Sources  
[Back to Top](#load-data-from-files-and-other-sources)  

In addition to loading data stored in files, <span>ML.NET</span> supports loading data from sources that include but are not limited to:
* In-memory collections
* JSON/XML
* Databases

Note that when working with streaming sources, <span>ML.NET</span> expects input to be in the form of an in-memory collection. Therefor when working with sources like JSON/XML, make sure to format the data into an in-memory collection.

Given the following in-memory collection:

```cs
HousingData[] inMemoryCollection = new HousingData[]
{
    new HousingData
    {
        Size = 700f,
        HistoricalPrices = new float[]
        {
            100000f, 3000000f, 250000f
        },
        CurrentPrice = 500000f
    },
    {
        Size = 1000f,
        HistoricalPrices = new float[]
        {
            600000f, 400000f, 650000f
        },
        CurrentPrice = 700000f
    }
};
```

Load the in-memory collection into an [IDataView](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.idataview) with the [LoadFromEnumerable](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.dataoperationscatalog.loadfromenumerable) method:

```cs
MLContext mlContext = new MLContext();

IDataView data = mlContext.Data
    .LoadFromEnumerable<HousingData>(inMemoryCollection);
```

> `LoadFromEnumerable` assumes that the `IEnumerable` it loads from is thread-safe.
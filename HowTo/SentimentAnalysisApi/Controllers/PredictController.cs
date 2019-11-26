using System;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using SentimentAnalysisApi.DataModels;

namespace SentimentAnalysisApi.Controllers
{
    [Route("api/[controller]")]
    public class PredictController : ControllerBase
    {
        private PredictionEnginePool<SentimentData, SentimentPrediction> engine;

        public PredictController(PredictionEnginePool<SentimentData, SentimentPrediction> engine)
        {
            this.engine = engine;
        }

        [HttpPost("[action]")]
        public ActionResult<string> PredictSentiment([FromBody]SentimentData input)
        {
            if (!ModelState.IsValid)
            {
                return BadRequest();
            }

            SentimentPrediction prediction = engine.Predict(input);

            string sentiment = Convert.ToBoolean(prediction.Prediction) ?
                "Positive" :
                "Negative";
            
            return Ok(sentiment);
        }
    }
}
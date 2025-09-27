import json
import logging

import azure.functions as func
import pandas as pd

from .modules import scikit_method, Faiss

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="scikit-learn")
def scikit(req: func.HttpRequest) -> func.HttpResponse:
   return func.HttpResponse("scikit-learn deduplication method.", status_code=200)


def _df_to_records(df: pd.DataFrame) -> list[dict]:
   """Return ISO-friendly records for API responses."""
   if df is None or df.empty:
      return []
   return json.loads(df.to_json(orient="records", date_format="iso"))


@app.route(route="faiss", methods=["POST"])
def faiss(req: func.HttpRequest) -> func.HttpResponse:
   try:
      payload = req.get_json()
   except ValueError:
      logging.exception("Failed to parse JSON payload for faiss route.")
      return func.HttpResponse("Invalid JSON body.", status_code=400)

   if isinstance(payload, dict):
      records = payload.get("data") or payload.get("records")
   else:
      records = payload

   if not isinstance(records, list):
      return func.HttpResponse(
         "Request body must contain a list of records under 'data'.",
         status_code=400,
      )

   try:
      df = pd.DataFrame(records)
   except Exception:
      logging.exception("Failed to construct DataFrame from payload.")
      return func.HttpResponse("Unable to build DataFrame from provided records.", status_code=400)

   required_cols = {"company", "title", "description", "job_date", "source", "job_id"}
   missing_cols = [col for col in required_cols if col not in df.columns]
   if missing_cols:
      return func.HttpResponse(
         f"Missing required columns: {', '.join(missing_cols)}.",
         status_code=400,
      )

   try:
      dedup_df, keep_latest = Faiss(df).dedup()
   except Exception:
      logging.exception("Faiss deduplication failed.")
      return func.HttpResponse("An error occurred during FAISS deduplication.", status_code=500)

   response_payload = {
      "dedup": _df_to_records(dedup_df),
      "keep_latest": _df_to_records(keep_latest),
   }

   return func.HttpResponse(
      json.dumps(response_payload),
      mimetype="application/json",
      status_code=200,
   )
